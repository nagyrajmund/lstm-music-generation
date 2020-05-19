import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.utils import rnn
from torch import optim
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from data.datasets import ClaraDataset
from data import utils
from torch.utils.data.sampler import SubsetRandomSampler

# TODO other dropout methods? e.g. variational dropout
# TODO implement windowed bptt

class AWD_LSTM(LightningModule):
    """
    AWD_LSTM that uses the following optimization strategies: temporal activation regularization, weight dropping,
    variable length backpropagation sequences and ASGD (optional).
    """

    def __init__(self, hparams):
        """
        Initialize network.
        Parameters contained in hparams:
            input_size:  input size
            embedding_size:  embedding size
            hidden_size:  hidden size; size of input and output in intermediate layers
            nlayers:  number of layers
            bias:  if True, use bias
            dropout_wts:  dropout rate
            asgd:  if True, use ASGD

        Parameters:
            hparams:  command-line arguments, see add_model_specific_args() for details
        """
        super().__init__()
        #TODO we have to load the dataset to get the number of tokens
        self.hparams = hparams
        self.dataset = ClaraDataset(hparams.dataset_path)

        self.embedding = nn.Embedding(self.dataset.n_tokens, hparams.embedding_size)

        # Layers #TODO: is batch_first = True ok?
        self.layers = self.construct_LSTM_layers()

        # Decoder
        self.decoder = nn.Linear(hparams.embedding_size, self.dataset.n_tokens)
    

    # ---------------------------- Model parameters ----------------------------

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--chunk_size', type=int, default=16)
        # If stride is not given, it's set to chunk_size to produce non-overlapping windows.
        parser.add_argument('--stride', type=int, nargs='?') 
        parser.add_argument('--embedding_size', type=int, default=400)
        parser.add_argument('--hidden_size', type=int, default=600)
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--use_asgd', type=bool, default=True)
        parser.add_argument('--use_bias', type=bool, default=True)
        parser.add_argument('--use_weight_dropout', type=bool, default=False)
        parser.add_argument('--p_dropout', type=float, default=0.5)
        parser.add_argument('--num_workers', type=int, default=1)
        return parser

    # ---------------------------- Initialisation ----------------------------
    
    def construct_LSTM_layers(self):
        LSTM_layers = []
        for i in range(self.hparams.n_layers):
            input_size = self.hparams.embedding_size if i == 0 else self.hparams.hidden_size
            output_size = self.hparams.hidden_size if i < (self.hparams.n_layers - 1) else self.hparams.embedding_size
            layer = nn.LSTM(input_size, output_size, bias=self.hparams.use_bias, batch_first = True)
            if self.hparams.use_weight_dropout:
                layer = WeightDropout(layer, self.hparams.p_dropout)
            
            LSTM_layers.append(layer)

        return nn.ModuleList(LSTM_layers)   

    def prepare_data(self):
         # TODO: This is a hacked version for debug runs.
         # We will implement the dataset splitting later.        
        
        self.train_dataset = self.dataset #, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

    def init_hidden(self, layer_hidden_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        h_init = torch.randn(1, self.hparams.batch_size, layer_hidden_size, device=self.device)
        c_init = torch.randn(1, self.hparams.batch_size, layer_hidden_size, device=self.device)

        h_init = torch.autograd.Variable(h_init)
        c_init = torch.autograd.Variable(c_init)

        return (h_init, c_init)

    # ---------------------------- Forward pass and learning steps ----------------------------

    def forward(self, X, X_lens): 
        """
        Forward pass.

        Parameters:
            x: input (batched)
            h: hidden state vector from previous activation (previous hidden state vector and cell state vector).
               Dimension: ((1, 1, input_size), (1, 1, input_size))

        Returns: hidden state vector and cell state vector as a tuple
        """
        self.batch_size, seq_len = X.size()
        print('X shape:', X.shape)
        X = self.embedding(X) # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        print('X after embedding', X.shape)
    
        idx = 0 # Iterate over chunks from all the sequences at the same time. 
        output = torch.empty((self.batch_size, seq_len, self.hparams.embedding_size))
        print('Output shape', output.shape)
        print(X)
        #TODO: how do I handle the X_lens argument for packing? i.e. if pack before the loop, how do we select the indices?
        #      if we pack the chunks in the loop, how do we select the indices still?
        while idx < seq_len - self.hparams.chunk_size: 
            X_chunk = X[:, idx:idx + self.hparams.chunk_size]     
            X_chunk = rnn.pack_padded_sequence(X_chunk, X_lens, batch_first=True, enforce_sorted=False) #TODO enforce sorted
            print('X after packing', X)

            initial_hiddens = construct_initial_hiddens()                
            
            layer_input = X_chunk
            for idx, LSTM_layer in enumerate(self.layers):
                layer_output, (h, c) = LSTM_layer(layer_input, initial_hiddens[idx])
                layer_input = layer_output

            layer_output, _ = rnn.pad_packed_sequence(layer_output, batch_first=True)

            network_output[:, idx:idx+self.hparams.chunk_size, :] = layer_output 
        # need to reshape the data so it goes into the linear layer
        
        output = output.contiguous().view(-1, self.hparams.embedding_size)
        output = self.decoder(output)

        # 4. Create log_softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, 0)
        output = F.log_softmax(output, dim=1)
        
        output = output.view(self.batch_size, seq_len,  self.dataset.n_tokens)
        print(output.shape)
        # TODO: detach?
        return output

    # ------------------------------------------------------------------------------------

    def construct_initial_hiddens(self):
        initial_hiddens = [self.init_hidden(self.hparams.hidden_size) for _ in range(self.hparams.n_layers - 1)]
        # The last layer's hidden size is the embedding size!
        initial_hiddens.append(self.init_hidden(self.hparams.embedding_size)) 
        return initial_hiddens
        
    # ------------------------------------------------------------------------------------    

    def configure_optimizers(self):
        if self.hparams.use_asgd:
            return optim.ASGD(self.parameters(), lr=self.hparams.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    # ------------------------------------------------------------------------------------

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, 
                          collate_fn=utils.pad_sequences, num_workers=self.hparams.num_workers)

    # ------------------------------------------------------------------------------------

    def general_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch
        output = self.forward(x, x_lens)
        loss = self.loss(output, y) 
        return loss

    # ------------------------------------------------------------------------------------

    def loss(self, prediction, labels):
        
        """ pred:   (batch_size, seq_len, n_tokens)
            labels: (batch_size, seq_len)
            
        """
        #1ST: expand prediction

        prediction = prediction.view(-1, self.dataset.n_tokens)
        # print("shape after flatten: ", prediction.shape)

        #2ND: map labels to tokens
        labels = labels.view(-1)


        #3RD: compute ce loss
        
        #3.1 create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = 0
        mask = (labels > tag_pad_token).float()
        
        #3.2 count how many tokens we have
        nb_tokens = int(np.sum(mask.cpu().numpy()))

        #3.3 pick the values for the label and zero out the rest with the mask
        #TODO very tricky type conversions here, better methods?
        prediction = prediction[range(int(list(prediction.size())[0])), labels] * mask

        #4TH compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(prediction) / nb_tokens

        return ce_loss

    # ------------------------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        tensorboard_logs = {'loss' : loss}
        
        return {'loss': loss, 'log': tensorboard_logs}

    # ---------------------------- Generate new music ----------------------------


    def generate(self, random_seed=0, input_len=10, predic_len=10):
        
        '''

        Parameters: 
            random_seed : seed to generate random input sequence
            input_len   : length of input sequence
            predic_len  : length of predicted sequence

        Flow:
            feed input tokens [a, b, c, d]
            pass forward to get output tokens [b, c, d, e]
            prediction append(e)

            feed input tokens[b, c, d, e] 
            pass forward to get output tokens[c, d, e, f]
            prediction append(f)

            ...

            return prediction [e, f, g, h, ..., ] of predic_len

        '''
        # set random seed
        torch.manual_seed(random_seed)

        # generate input sequence, randomly sample from dataset.n_tokens
        input_seq = torch.randint(0, self.dataset.n_tokens - 1, (input_len,))
        
        # feed LongTensor to LSTM
        layer_input = torch.unsqueeze(torch.LongTensor(input_seq), 0)

        # Embeddiing
        layer_input = self.embedding(layer_input)
        
        predicted = []
        

        for i in range(predic_len):
            
            initial_hiddens = [self.init_hidden(self.hparams.hidden_size) for _ in range(self.hparams.n_layers - 1)]
            initial_hiddens.append(self.init_hidden(self.hparams.embedding_size))
            
            for idx, LSTM_layer in enumerate(self.layers):
                output, (h, c) = LSTM_layer(layer_input, initial_hiddens[idx])
                layer_input = output
            
            output = self.decoder(output)
            output = F.log_softmax(output, dim=1)
            output = torch.argmax(output, dim=2)
            predicted.append(output[0][-1].item())
            layer_input = self.embedding(output)

        return predicted

    # ------------------------------------------------------------------------------------
        

class WeightDropout(nn.Module):
    """ Adapted from original salesforce paper. """
    def __init__(self, module: nn.Module, p_dropout: float = 0.5): # Default value of p_dropout is 0.5 in the original paper!
        print("Using weight dropout!")
        super().__init__()
        self.module = module
        self.hparams_dropout = p_dropout
        
        w = self.module.weight_hh_l0
        # del self.module._parameters['weight_hh_l0']
        
        self.register_parameter('weight_raw', nn.Parameter(w.data))
        
    def _setweights(self):
        "Apply dropout to the raw weights."
        raw_w = self.weight_raw #TODO check if ok
        self.module._parameters['weight_hh_l0'] = F.dropout(raw_w, p=self.hparams_dropout, training=self.training) #TODO check if training is passed correctly

    def forward(self, input, hiddens):
        self._setweights()
        return self.module(input, hiddens)
