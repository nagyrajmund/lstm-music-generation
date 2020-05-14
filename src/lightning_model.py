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
from torchsummary import summary
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
        self.module.weight_hh_l._parameters['weight'] = F.dropout(raw_w, p=self.weight_p, training=self.training) #TODO check if training is passed correctly

    def forward(self):
        self._setweights()
        return self.module.forward()


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
    
        self.embedding = nn.Embedding(self.train_dataset.n_tokens, P.embedding_size)

        # Layers #TODO: is batch_first = True ok?
        self.layers = nn.ModuleList([
            WeightDropout(nn.LSTM(P.embedding_size, P.hidden_size,    bias=P.use_bias, batch_first=True), P.p_dropout),
            WeightDropout(nn.LSTM(P.hidden_size,    P.hidden_size,    bias=P.use_bias, batch_first=True), P.p_dropout),
            WeightDropout(nn.LSTM(P.hidden_size,    P.embedding_size, bias=P.use_bias, batch_first=True), P.p_dropout)
        ])

        # Decoder
        self.decoder = nn.Linear(P.embedding_size, self.train_dataset.n_tokens)

    def forward(self, x, hiddens):
        """
        Forward pass.

        Parameters:
            x: input (batched)
            h: hidden state vector from previous activation (previous hidden state vector and cell state vector).
               Dimension: ((1, 1, input_size), (1, 1, input_size))

        Returns: hidden state vector and cell state vector as a tuple
        """
        x_in, _ = batch
        h_in, c_in = hiddens
        
        #DIRTY HACK: unpack x, embed the elements then pack it back for the LSTM
        h_in = rnn.PackedSequence(self.embedding(x_in.data), x_in.batch_sizes)

        for LSTM_layer in self.layers:
            output, h_in, c_in = LSTM_layer(h_prev, (h_in, c_in))
        
        batchsize, inputsize, embedsize = output.size
        output = output.detach()
        output = output.view(batchsize * inputsize, outputsize)
        output = self.decoder(output)
        output = output.view(batchsize, inputsize, outputsize)
       
        return output, (h_in, c_in)
        
    def configure_optimizers(self):
        if self.P.asgd:
            return optim.ASGD(self.parameters(), lr=self.P.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        return optim.Adam(self.parameters(), lr=self.P.lr)

    def prepare_data(self):
        #TODO load vocab dicts if there are any
        pass
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.P.batch_size, collate_fn=rnn.pack_sequence)
        
    # def val_dataloader(self):
    #     # TODO return DataLoader
    #     pass

    # def test_dataloader(self):
    #     # TODO return DataLoader
    #     pass

    def general_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        tensorboard_logs = {'loss' : loss}
        return {'loss': loss, 'log': tensorboard_logs}

    # def validation_step(self, batch, batch_idx):
    #     loss = self.general_step(batch, batch_idx)
    #     return {'val_loss': loss}

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
    parser.add_argument('--dataset_path', type=str, default='../dataset/piano_solo/note_range38/sample_freq4/jazz/')
    parser = AWD_LSTM.add_model_specific_args(parser) # Add model-specific args
    parser = Trainer.add_argparse_args(parser) # Add ALL training-specific args
    return parser

if __name__ == "__main__":
    # Parse command-line args
    hparams = build_argument_parser().parse_args()
    model = AWD_LSTM(hparams)
    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model)
