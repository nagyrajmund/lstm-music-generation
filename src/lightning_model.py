import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from network.awd_lstm import WeightDropout
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

class AWD_LSTM(LightningModule):
    """
    AWD_LSTM that uses the following optimization strategies: temporal activation regularization, weight dropping,
    variable length backpropagation sequences. Additional methods may be employed from the outside, e.g. ASGD.
    """
    # TODO other dropout methods? e.g. variational dropout
    # TODO implement windowed bptt

    def __init__(self, hparams):
        """
        Initialize network.
        Parameters contained in hparams:
            ntokens:  number of tokens
            input_size:  input size
            embedding_size:  embedding size
            hidden_size:  hidden size; size of input and output in intermediate layers
            nlayers:  number of layers
            bias:  if True, use bias
            device:  device
            dropout_wts:  dropout rate

        Parameters:
            hparams:  command-line arguments, see add_model_specific_args() for details
        """
        
        super(AWD_LSTM, self).__init__()
        
        self.ntokens = hparams.ntokens
        self.nlayers = hparams.nlayers
        self.hidden_size = hparams.hidden_size
        self.device = hparams.device
        self.embedding_size = hparams.embedding_size

        # Embedding
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)

        # Layers
        self.layers = []
        for i in range(self.nlayers):
            if i == 0:
                cell_input_s = self.embedding_size
            else:
                cell_input_s = self.hidden_size

            if i == self.nlayers - 1:
                cell_output_s = self.embedding_size
            else:
                cell_output_s = self.hidden_size

            layer = nn.LSTM(cell_input_s, cell_output_s, bias=hparams.bias)
            layer = WeightDropout(layer, hparams.dropout_wts) # Weight dropping
            self.layers.append(layer)

        # Decoder
        self.decoder = nn.Linear(embedding_size, self.ntokens)
        
        self.output = None

    def forward(self, x, hiddens):
        """
        Forward pass.

        Parameters:
            x: input vector of size (batch_size, 1, input_size)
            h: hidden state vector from previous activation (previous hidden state vector and cell state vector).
               Dimension: ((1, 1, input_size), (1, 1, input_size))

        Returns: hidden state vector and cell state vector as a tuple
        """
        
        h, c = hiddens
        output = Tensor().to(self.device)
        
        # Propagate through layers for each timestep
        for t in range(x.size(0)):    
            
            inp = x[t, :, :]
            h_prev = inp
            h_new = []
            c_new = []

            for i in range(self.nlayers):
                h_curr, c_curr = self.layer[i](h_prev, (h[i], c[i]))
                hprev = h[i]
                h_new.append(h_curr)
                c_new.append(c_curr)

            h = h_new # (1, 1, current_output_size)
            c = c_new # (1, 1, current_output_size)
            
            # TAR - temporal activation regularization
            output = th.cat((output, h[-1].unsqueeze(0)))
            
        self.output = output.detach() # Dimension is (batch_size, 1, embedding_size)
        
        # Translate embedding vectors to tokens
        reshaped = output.view(output.size(0) * output.size(1), output.size(2)) # (batch_size, embedding_size)
        decoded = self.decoder(reshaped) # (batch_size, ntokens)
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1)) # (batch_size, ntokens, ntokens)

        return decoded, (h, c)
        
    def configure_optimizers(self):
        return optim.ASGD(self.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

    def prepare_data(self):
        # prepare Dataset
        pass 

    def train_dataloader(self):
        # return DataLoader
        pass

    def val_dataloader(self):
        # return DataLoader
        pass

    def test_dataloader(self):
        # return DataLoader
        pass

    def general_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        tensorboard_logs = {'loss' : loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {'test_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_epochs', type=int, default=20)
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument('--embedding_size', type=int, default=400)
        parser.add_argument('--hidden_size', type=int, default=600)
        parser.add_argument('--nlayers', type=int, default=4)
        parser.add_argument('--lr', type=int, default=0.0001)
        return parser

def build_argument_parser():
    parser = ArgumentParser()
    parser = AWD_LSTM.add_model_specific_args(parser) # Add model-specific args
    parser = Trainer.add_argparse_args(parser) # Add ALL training-specific args
    return parser

if __name__ == "__main__":
    # Parse command-line args
    hparams = build_argument_parser().parse_args()
    model = AWD_LSTM(hparams)
    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model)
    trainer.test()