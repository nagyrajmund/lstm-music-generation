import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from torch import optim
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from reworked_dataset import ClaraDataset
from data import utils

class WeightDropout(nn.Module):
    """ Adapted from original salesforce paper. """
    def __init__(self, module: nn.Module, p_dropout: float = 0.5): # Default value of p_dropout is 0.5 in the original paper!
        super().__init__()
        self.module = module
        self.p_dropout = p_dropout
        
        w = self.module.weight_hh_l0
        # del self.module._parameters['weight_hh_l0']
        
        self.register_parameter('weight_raw', nn.Parameter(w.data))
        

    def _setweights(self):
        "Apply dropout to the raw weights."
        raw_w = self.weight_raw #TODO check if ok
        self.module._parameters['weight_hh_l0'] = F.dropout(raw_w, p=self.p_dropout, training=self.training) #TODO check if training is passed correctly

    def forward(self, input, hiddens):
        self._setweights()
        print('[LOG] weight dropout done')
        return self.module(input, hiddens)


class AWD_LSTM(LightningModule):
    """
    AWD_LSTM that uses the following optimization strategies: temporal activation regularization, weight dropping,
    variable length backpropagation sequences and ASGD (optional).
    """
    # TODO other dropout methods? e.g. variational dropout
    # TODO implement windowed bptt

    def __init__(self, P):
        """
        Initialize network.
        Parameters contained in hparams:
            input_size:  input size
            embedding_size:  embedding size
            hidden_size:  hidden size; size of input and output in intermediate layers
            nlayers:  number of layers
            bias:  if True, use bias
            device:  device
            dropout_wts:  dropout rate
            asgd:  if True, use ASGD

        Parameters:
            hparams:  command-line arguments, see add_model_specific_args() for details
        """
        super().__init__()
        #TODO we have to load the dataset to get the number of tokens
        self.P = hparams

        self.train_dataset = ClaraDataset(P.dataset_path)
        print('[LOG] created dataset!')
        self.embedding = nn.Embedding(self.train_dataset.n_tokens, P.embedding_size)

        # Layers #TODO: is batch_first = True ok?
        self.layers = nn.ModuleList([
            WeightDropout(nn.LSTM(P.embedding_size, P.hidden_size,    bias=P.use_bias, batch_first=True), P.p_dropout),
            WeightDropout(nn.LSTM(P.hidden_size,    P.hidden_size,    bias=P.use_bias, batch_first=True), P.p_dropout),
            WeightDropout(nn.LSTM(P.hidden_size,    P.embedding_size, bias=P.use_bias, batch_first=True), P.p_dropout)
        ])

        # Decoder
        self.decoder = nn.Linear(P.embedding_size, self.train_dataset.n_tokens)

    def init_hidden(self, layer_hidden_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        #TODO: correct device pls
        h_init = torch.randn(1, self.P.batch_size, layer_hidden_size)#, device=self.device)
        c_init = torch.randn(1, self.P.batch_size, layer_hidden_size)#, device=self.device)

        h_init = torch.autograd.Variable(h_init)
        c_init = torch.autograd.Variable(c_init)

        return (h_init, c_init)

    def forward(self, X, X_lens):
        """
        Forward pass.

        Parameters:
            x: input (batched)
            h: hidden state vector from previous activation (previous hidden state vector and cell state vector).
               Dimension: ((1, 1, input_size), (1, 1, input_size))

        Returns: hidden state vector and cell state vector as a tuple
        """
        print("input shape before embedding: ", X.size())
        #TODO: Do we want integrate the size into models? e.g. self.batch_size
        self.batch_size, seq_len = X.size()

        print('[LOG] doing forward!')
        print('[LOG] embedding...')
        X = self.embedding(X) # embedding takes the padded sequence, and not the packed_and_padded sequence because it cannot operate on the latter
        print('[LOG] packing...')

        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = rnn.pack_padded_sequence(X, X_lens, batch_first=True, enforce_sorted=False) #TODO enforce sorted
         
        layer_hidden_sizes = [self.P.hidden_size, self.P.hidden_size, self.P.embedding_size]
        initial_hiddens = [self.init_hidden(hidden_size) for hidden_size in layer_hidden_sizes]

        layer_input = X
        for idx, LSTM_layer in enumerate(self.layers):
            output, (h, c) = LSTM_layer(layer_input, initial_hiddens[idx])
            layer_input = output


        X, _ = rnn.pad_packed_sequence(output, batch_first=True)
        
        # X.shape is (batch_size, seq_len, embedding_size)

        # need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # Linear mapping
        X = self.decoder(X)
        print("shape after decoding")
        print(X.shape)
        self.nb_tags = X.shape[1]

        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        X = F.log_softmax(X, dim=1)

        X = X.view(self.batch_size, seq_len, self.nb_tags)
        # TODO: detach?
       
        return X
        
    def configure_optimizers(self):
        if self.P.asgd:
            return optim.ASGD(self.parameters(), lr=self.P.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        return optim.Adam(self.parameters(), lr=self.P.lr)

    def prepare_data(self):
        #TODO load vocab dicts if there are any
        pass
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.P.batch_size, collate_fn=utils.pad_sequences)

    #TODO: the validation dataloader is only here so lightning can run fast_dev_run
    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.P.batch_size, collate_fn=utils.pad_sequences)
        
    # def val_dataloader(self):
    #     # TODO return DataLoader
    #     pass

    # def test_dataloader(self):
    #     # TODO return DataLoader
    #     pass

    def general_step(self, batch, batch_idx):
        print('New batch!')
        x, y, x_lens, y_lens = batch
        output = self.forward(x, x_lens)
        print("output shape: ", output.shape)
        print("y shape: ", y.shape)
        print("output: ")
        print(output)
        print("y: ")
        print(y)
        loss = self.loss(output, y) # TODO @Logan: make sure the lsos function can eat Y because it's padded.
        return loss

    def loss(self, prediction, labels):
        """ pred:   (batch_size, seq_len, n_tokens)
            labels: (batch_size, seq_len)
        """
        


        # TODO @Logan: filter out the paddings so that they don't influence the loss 
        # i.e. x: [0, 1, 2, 6, PAD, PAD, PAD], y: [1,2,6,5,PAD,PAD,PAD] -> operate on x[:4] and y[:4] only
        # see this for details https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        F.cross_entropy(prediction.view(-1, ), labels)

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        tensorboard_logs = {'loss' : loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)


    # def test_step(self, batch, batch_idx):
    #     loss = self.general_step(batch, batch_idx)
    #     return {'test_loss': loss}
    
    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'test_loss': avg_loss}
    #     return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--embedding_size', type=int, default=400)
        parser.add_argument('--hidden_size', type=int, default=600)
        parser.add_argument('--lr', type=int, default=0.0001)
        parser.add_argument('--asgd', type=bool, default=True)
        parser.add_argument('--use_bias', type=bool, default=True) #TODO
        parser.add_argument('--p_dropout', type=float, default=0.5)
        return parser

def build_argument_parser():
    parser = ArgumentParser()
    #TODO: chagne back to relative path (temp fix for debugger)
    parser.add_argument('--dataset_path', type=str, default=r'C:\Users\User\Desktop\clara\datasets\chordwise\chamber\note_range38\sample_freq4\debussy')
    parser = AWD_LSTM.add_model_specific_args(parser) # Add model-specific args
    parser = Trainer.add_argparse_args(parser) # Add ALL training-specific args
    return parser

if __name__ == "__main__":
    # Parse command-line args
    hparams = build_argument_parser().parse_args()
    model = AWD_LSTM(hparams)
    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model)
