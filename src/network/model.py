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
from torch.autograd import Variable
from tqdm import tqdm

# TODO embedding dropout
# TODO implement variational length backprop

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
        self.dataset = ClaraDataset(hparams.dataset_path, chunk_size=hparams.chunk_size)

        self.embedding = nn.Embedding(self.dataset.n_tokens, hparams.embedding_size)
        # TODO self.vd = VariationalDropout()
        self.layers = self.construct_LSTM_layers()
        self.decoder = nn.Linear(hparams.embedding_size, self.dataset.n_tokens)

        if not torch.cuda.is_available():
            self.device = "cpu"

    # ---------------------------- Model parameters ----------------------------

    @staticmethod
    def add_model_specific_args(parent_parser):
        #TODO: boolean arguments don't work as expected, watch out!
        #      if you pass any value to them, e.g `python train.py use_bias --False` then use_bias will be set to True!
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--chunk_size', type=int, default=1, help='chunk size')
        parser.add_argument('--save_interval', type=int, default=100, help='checkpoint saving interval (in epochs)')
        # If stride is not given, it's set to chunk_size to produce non-overlapping windows.
        # parser.add_argument('--stride', type=int, nargs='?') 
        parser.add_argument('--embedding_size', type=int, default=600, help='embedding size') #todo def value
        parser.add_argument('--hidden_size', type=int, default=400, help='hidden size') #todo def value
        parser.add_argument('--n_layers', type=int, default=1, help='number of LSTM layers')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--use_asgd', type=bool, default=False, help='use ASGD')
        parser.add_argument('--use_bias', type=bool, default=True, help='use bias')
        parser.add_argument('--use_weight_dropout', type=bool, default=False, help='use weight dropout')
        parser.add_argument('--p_dropout', type=float, default=0.5, help='dropout rate for weight dropout')
        parser.add_argument('--dropouti', type=float, default=0.5, help='dropout rate in input layer (for variational dropout)')
        parser.add_argument('--dropouth', type=float, default=0.5, help='dropout rate in hidden layer (for variational dropout)')
        parser.add_argument('--dropouto', type=float, default=0.5, help='dropout rate in output layer (for variational dropout)')
        parser.add_argument('--alpha', type=float, default=0, help='coefficient for activation regularization')
        parser.add_argument('--beta', type=float, default=0, help='coefficient for temporal activation regularization')
        parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
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
        h_init = torch.autograd.Variable(torch.randn(1, self.hparams.batch_size, layer_hidden_size, device=self.device))
        c_init = torch.autograd.Variable(torch.randn(1, self.hparams.batch_size, layer_hidden_size, device=self.device))

        return (h_init, c_init)

    # ---------------------------- Forward pass and learning steps ----------------------------

    def forward(self, X_in):
        """
        Forward pass.

        Parameters:
            x: input (batched)
            h: hidden state vector from previous activation (previous hidden state vector and cell state vector).
               Dimension: ((1, 1, input_size), (1, 1, input_size))

        Returns: hidden state vector and cell state vector as a tuple
        """
        X_in = torch.squeeze(X_in, dim=0)
        # -> X_in: (batch_size, n_chunks, chunk_size)
        
        batch_size, n_chunks, chunk_size = X_in.size()
        assert(chunk_size == self.hparams.chunk_size)
        assert(batch_size == self.hparams.batch_size)

        X_in = self.embedding(X_in)
        # -> X_in: (batch_size, n_chunks, chunk_size, embedding_size)
        # TODO X_in = self.vd(X_in, self.hparams.dropouti) # Variational dropout
        all_outputs = torch.empty(batch_size, n_chunks, chunk_size, self.hparams.embedding_size, device=self.device)
        # -> all_outputs: (batch_size, n_chunks, chunk_size, embedding_size)
        for chunk_idx in range(chunk_size):
            chunk = X_in[:, chunk_idx]
            # -> chunk: (batch_size, chunk_size, embedding_size)
            initial_hiddens = self.construct_initial_hiddens()  
            for layer_idx, LSTM_layer in enumerate(self.layers):
                chunk, _ = LSTM_layer(chunk, initial_hiddens[layer_idx])
                # -> chunk: (batch_size, chunk_size, embedding_size)

                # Variational dropout TODO
                # if layer_idx == self.hparams.n_layers:
                #     chunk = self.vd(chunk, self.hparams.dropouth)
                # else:
                #     chunk = self.vd(chunk, self.hparams.dropouto)

            all_outputs[:, chunk_idx] = chunk

        all_outputs = self.decoder(all_outputs)
        # -> all_outputs: (batch_size, n_chunks, chunk_size, n_tokens)

        # permute the outputs for the cross_entropy loss later
        return all_outputs.permute(0, 3, 1, 2)
        # -> all_outputs: (batch_size, n_tokens, n_chunks, chunk_size)
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
                          num_workers=self.hparams.num_workers)

    # ------------------------------------------------------------------------------------

    def general_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        y = torch.squeeze(y, dim=0)
        loss = F.cross_entropy(output, y)

        # Activation regularization # TODO check
        if self.hparams.alpha:
            loss += self.hparams.alpha * sum(output[:, :, :, i].pow(2).mean() for i in range(self.hparams.chunk_size))

        # Temporal activation regularization
        if self.hparams.beta:
            diff = output[:, :, :, 1:] - output[:, :, :, :-1]
            loss += self.hparams.beta * sum(diff[:, :, :, i].pow(2).mean() for i in range(self.hparams.chunk_size - 1))
        return loss

    # ------------------------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        tensorboard_logs = {'loss' : loss}
        
        return {'loss': loss, 'log': tensorboard_logs}

    # ------------------------------------------------------------------------------------

    def training_epoch_end(self, outputs):
        if (self.trainer.current_epoch != 0) and (self.trainer.current_epoch % self.hparams.save_interval == 0):
            # Save state dict with parameters (checkpoint)
            model_data = {'state_dict': self.state_dict(), 'hparams': self.hparams}
            model_full_path = \
                self.hparams.model_path + "/" + self.hparams.model_file + "_" + str(self.trainer.current_epoch) + "epochs.pth"
            torch.save(model_data, model_full_path)

        # Average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}
        return {'avg_train_loss': avg_loss, 'log': tensorboard_logs}

    # ------------------------------------------------------------------------------------

    def on_train_end(self):
        # Save state dict with parameters at the end of the training
        model_data = {'state_dict': self.state_dict(), 'hparams': self.hparams}
        torch.save(model_data, self.hparams.model_path + "/" + self.hparams.model_file + ".pth")

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
        input_seq = torch.randint(0, self.dataset.n_tokens - 1, (input_len,), device=self.device)
        
        # feed LongTensor to LSTM
        layer_input = torch.unsqueeze(torch.LongTensor(input_seq), 0)

        # Embeddiing
        layer_input = self.embedding(layer_input)
        
        predicted = [1]
        

        for i in tqdm(range(predic_len)):
            
            initial_hiddens = [self.init_hidden(self.hparams.hidden_size) for _ in range(self.hparams.n_layers - 1)]
            initial_hiddens.append(self.init_hidden(self.hparams.embedding_size))
            
            for idx, LSTM_layer in enumerate(self.layers):
                layer_output, (h, c) = LSTM_layer(layer_input, initial_hiddens[idx])
                layer_input = layer_output
            
            layer_output = self.decoder(layer_output)
            layer_output = F.log_softmax(layer_output, dim=1)
            layer_output = torch.argmax(layer_output, dim=2)
            layer_input = self.embedding(layer_output)

            output = layer_output.tolist()[0]
            while 0 in output: output.remove(0)
            while 1 in output: output.remove(1)
            while 2 in output: output.remove(2)   
            predicted.extend(output)

        predicted.append(3)
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

class VariationalDropout(nn.Module):
    """ Adapted from original salesforce paper. TODO rewrite logic """

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if dropout is None:
            return x

        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x