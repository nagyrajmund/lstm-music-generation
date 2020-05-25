import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
from utils.generate_helper import generate_sound
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

# TODO implement variational length backprop

class AWD_LSTM(LightningModule):
    """
    AWD_LSTM that uses the following optimization strategies: temporal activation regularisation, weight dropping,
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
        self.dataset = ClaraDataset(hparams.dataset_path, chunk_size=hparams.chunk_size, batch_size=hparams.batch_size)

        self.embedding = nn.Embedding(self.dataset.n_tokens, hparams.embedding_size)
        if self.hparams.use_embedding_dropout:
            self.embedding = EmbeddingDropout(self.embedding, self.hparams.dropoute)
        self.layers = self.construct_LSTM_layers()
        self.decoder = nn.Linear(hparams.embedding_size, self.dataset.n_tokens)

        if not hasattr(self, 'device'):
            if not torch.cuda.is_available():
                self.device = "cpu"
            else:
                self.device = "cuda:0"
    # ---------------------------- Model parameters ----------------------------

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        parser.add_argument('--chunk_size', type=int, default=4, help='chunk size')
        parser.add_argument('--save_interval', type=int, default=100, help='checkpoint saving interval (in epochs)')
        # If stride is not given, it's set to chunk_size to produce non-overlapping windows.
        # parser.add_argument('--stride', type=int, nargs='?') 
        parser.add_argument('--embedding_size', type=int, default=600, help='embedding size')
        parser.add_argument('--hidden_size', type=int, default=400, help='hidden size')
        parser.add_argument('--n_layers', type=int, default=1, help='number of LSTM layers')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--use_asgd', action='store_true', default=False, help='use ASGD')
        parser.add_argument('--without_bias', action='store_true', default=False, help='do not use bias')
        parser.add_argument('--without_weight_dropout', action='store_true', default=False, help='do not use weight dropout')
        parser.add_argument('--p_dropout', type=float, default=0.5, help='dropout rate for weight dropout')
        parser.add_argument('--use_variational', action='store_true', default=False, help='use variational dropout')
        parser.add_argument('--dropouti', type=float, default=0.5, help='dropout rate in input layer (for variational dropout)')
        parser.add_argument('--dropouth', type=float, default=0.5, help='dropout rate in hidden layer (for variational dropout)')
        parser.add_argument('--dropouto', type=float, default=0.5, help='dropout rate in output layer (for variational dropout)')
        parser.add_argument('--use_embedding_dropout', action='store_true', default=False, help='use embedding dropout')
        parser.add_argument('--dropoute', type=float, default=0.5, help='dropout rate in embedding matrix (for embedding dropout)')
        parser.add_argument('--alpha', type=float, default=0, help='coefficient for activation regularisation')
        parser.add_argument('--beta', type=float, default=0, help='coefficient for temporal activation regularisation')
        parser.add_argument('--without_scaled_loss', action='store_true', default=False, help='do not use scaled loss')
        parser.add_argument('--topk', type=int, default=1, help='sample from the top k predictions')
        parser.add_argument('--sampling_freq', type=int, default=0.5, help='frequency of sampling from the top k predictions')
        parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
        return parser

    # ---------------------------- Initialisation ----------------------------
    
    def construct_LSTM_layers(self):
        LSTM_layers = []
        for i in range(self.hparams.n_layers):
            input_size = self.hparams.embedding_size if i == 0 else self.hparams.hidden_size
            output_size = self.hparams.hidden_size if i < (self.hparams.n_layers - 1) else self.hparams.embedding_size
            layer = nn.LSTM(input_size, output_size, bias=(not self.hparams.without_bias), batch_first = True)
            if not self.hparams.without_weight_dropout: # Use WD
                layer = WeightDropout(layer, self.hparams.p_dropout)
            
            LSTM_layers.append(layer)

        return nn.ModuleList(LSTM_layers)   

    def prepare_data(self):
         # TODO: This is a hacked version for debug runs.
         # We will implement the dataset splitting later.        
        
        self.train_dataset = self.dataset #, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

    def init_hidden(self, layer_hidden_size, batch_size):
        h_init = torch.autograd.Variable(torch.randn(1, batch_size, layer_hidden_size, device=self.device))
        c_init = torch.autograd.Variable(torch.randn(1, batch_size, layer_hidden_size, device=self.device))

        return (h_init, c_init)

    # ---------------------------- Forward pass and learning steps ----------------------------

    def forward(self, X_in, is_training = True):
        """
        Forward pass.

        Parameters:
            x: input (batched)
            h: hidden state vector from previous activation (previous hidden state vector and cell state vector).
               Dimension: ((1, 1, input_size), (1, 1, input_size))

        Returns: hidden state vector and cell state vector as a tuple
        """
        batch_size, chunk_size = X_in.size()
        if is_training:
            assert(batch_size == self.hparams.batch_size)
            assert(chunk_size == self.hparams.chunk_size)
        else:
            assert(batch_size == 1)

        X_in = self.embedding(X_in)
        # -> X_in: (batch_size, chunk_size, embedding_size)
        X_in = self.variational_dropout(X_in, "input")
        
        initial_hiddens = self.construct_initial_hiddens(batch_size)  
        layer_input = X_in
        for layer_idx, LSTM_layer in enumerate(self.layers):
            output, _ = LSTM_layer(layer_input, initial_hiddens[layer_idx])
 
            if layer_idx < self.hparams.n_layers-1:
                layer_input = self.variational_dropout(output, "hidden")

        output = self.variational_dropout(output, "output")
        # output: (batch_size, chunk_size, embedding_size)
        output = self.decoder(output)
        # -> output: (batch_size, chunk_size, n_tokens)

        return output.permute(0,2,1)
        # -> output: (batch_size, n_tokens, chunk_size)

    def variational_dropout(self, x, mode):
        if not self.hparams.use_variational:
            return x
        
        if mode == 'input':
            p = self.hparams.dropouti # TODO: better command line parameter names
        elif mode == 'hidden':
            p = self.hparams.dropouth
        elif mode == 'output':
            p = self.hparams.dropouto
        else:
            print(f"Unknown dropout type {mode} in variational_dropout!")
            exit()
        
        return F.dropout(x, p)

    # ------------------------------------------------------------------------------------

    def construct_initial_hiddens(self, batch_size):
        initial_hiddens = [self.init_hidden(self.hparams.hidden_size, batch_size) for _ in range(self.hparams.n_layers - 1)]
        # The last layer's hidden size is the embedding size!
        initial_hiddens.append(self.init_hidden(self.hparams.embedding_size, batch_size)) 
        return initial_hiddens
        
    # ------------------------------------------------------------------------------------    

    def configure_optimizers(self):
        if self.hparams.use_asgd:
            return optim.ASGD(self.parameters(), lr=self.hparams.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    # ------------------------------------------------------------------------------------

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.hparams.num_workers)

    # ------------------------------------------------------------------------------------

    def general_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.squeeze(0), y.squeeze(0)
        output = self.forward(x)
        # -> output: (batch_size, n_tokens, chunk_size)

        # Wait penalisation
        if not self.hparams.without_scaled_loss:
            sum_of_waits = 0

            freq = torch.Tensor(list(self.dataset.token_count.values())).to(self.device)
            weights = self.dataset.n_tokens / freq
            loss = F.cross_entropy(output, y, weight=weights)
        else:
            loss = F.cross_entropy(output, y)

        # Activation regularisation 
        if self.hparams.alpha:
            loss += self.hparams.alpha * sum(output[:, :, i].pow(2).mean() for i in range(self.hparams.chunk_size))

        # Temporal activation regularisation
        if self.hparams.beta:
            diff = output[:, :, 1:] - output[:, :, :-1]
            loss += self.hparams.beta * sum(diff[:, :, i].pow(2).mean() for i in range(self.hparams.chunk_size - 1))

        return loss

    # ------------------------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        tensorboard_logs = {'loss' : loss}
        
        return {'loss': loss, 'log': tensorboard_logs}

    # ------------------------------------------------------------------------------------

    def training_epoch_end(self, outputs):
        # Save state dict with parameters (checkpoint)
        if (self.trainer.current_epoch != 0) and (self.trainer.current_epoch % self.hparams.save_interval == 0):
            model_data = {'state_dict': self.state_dict(), 'hparams': self.hparams}
            model_name = self.hparams.model_file + "_" + str(self.trainer.current_epoch) + "epochs"
            model_full_path = \
                self.hparams.model_path + "/" + model_name + ".pth"
            torch.save(model_data, model_full_path)

            # Generate
            args = copy.deepcopy(self.hparams)
            args.model_file = model_name
            generate_sound(self, args, use_tqdm=False)

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

    def generate(self, random_seed, input_len, predic_len, use_tqdm=True):
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
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # generate input sequence, randomly sample from dataset.n_tokens
        input_size = (1, self.hparams.chunk_size)
        input_seq = torch.randint(0, self.dataset.n_tokens - 1, input_size, device=self.device)

        # Forward pass
        predicted = []

        if use_tqdm:
            iter_range = tqdm(range(predic_len))
        else:
            iter_range = (range(predic_len))

        for i in iter_range:
            if random.random() < self.hparams.sampling_freq:
                k = int(self.hparams.topk) # TODO remove
            else:
                k = 1

            output = self.forward(input_seq, is_training=False) # (1, n_tokens, chunk_size)
            values, indices = torch.topk(output, k=k, dim=1) # (1, k, chunk_size)

            output = torch.empty((1, self.hparams.chunk_size), dtype=int, device=self.device)

            for j in range(self.hparams.chunk_size):
                ind = torch.multinomial(values[0, :, j], 1)
                output[0, j] = indices[0, ind, j]

            input_seq = output.clone().detach() # Input for the next iteration
            
            output = output.tolist()[0]
            predicted.extend(output)
        return predicted

    # ------------------------------------------------------------------------------------
        

class WeightDropout(nn.Module):
    """ Adapted from original salesforce paper. """
    def __init__(self, module: nn.LSTM, p_dropout: float = 0.5): # Default value of p_dropout is 0.5 in the original paper!
        print("Using weight dropout!")
        super().__init__()
        self.module = module
        self.hparams_dropout = p_dropout
        
        w = self.module.weight_hh_l0
        # del self.module._parameters['weight_hh_l0']
        
        self.register_parameter('weight_raw', nn.Parameter(w.data))
        
    def _setweights(self):
        "Apply dropout to the raw weights."
        raw_w = self.weight_raw
        self.module._parameters['weight_hh_l0'] = F.dropout(raw_w, p=self.hparams_dropout, training=self.training)
        self.module._parameters['weight_hh_l0'].retain_grad()

    def forward(self, input, hiddens):
        self._setweights()
        return self.module(input, hiddens)


class EmbeddingDropout(nn.Module):
    def __init__(self, module: nn.Embedding, dropoute: float = 0.5):
        print("Using embedding dropout!")
        super().__init__()
        self.module = module
        self.dropoute = dropoute
        
        w = self.module.weight
        self.register_parameter('weight_raw', nn.Parameter(w.data))
        
    def _setweights(self):
        "Apply dropout to the raw weights."
        raw_w = self.weight_raw
        self.module._parameters['weight'] = F.dropout(raw_w, p=self.dropoute, training=self.training)
        self.module._parameters['weight'].retain_grad()

    def forward(self, input):
        self._setweights()
        return self.module(input)