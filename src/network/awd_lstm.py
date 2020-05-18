import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor 

# A simplified version of AWD-LSTM implment from scratch, code adapted from https://github.com/a-martyn/awd-lstm

class WeightDropout(nn.Module):
    
    def __init__(self, module: nn.LSTMCell, weight_p: float):
        super().__init__()
        self.module, self.weight_p = module, weight_p
            
        w = getattr(self.module, 'weight_hh')
        self.register_parameter('weight_raw', nn.Parameter(w.data))
        self.module.weight_hh = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        
        "Apply dropout to the raw weights."
        raw_w = getattr(self, 'weight_raw')
        self.module.weight_hh = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self):
        self._setweights()
        return self.module.forward()

# @deprecated
class AWD_LSTM(nn.Module):
    """
    AWD_LSTM that uses the following optimization strategies: temporal activation regularization, weight dropping,
    variable length backpropagation sequences. Additional methods may be employed from the outside, e.g. ASGD.
    """

    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, nlayers: int=3, 
                 bias: bool=True, device: str='cpu', dropout_wts: float=0.5):
        """
        Initialize network.

        Parameters:
            input_size:  input size
            embedding_size:  embedding size
            hidden_size:  hidden size; size of input and output in intermediate layers
            nlayers:  number of layers
            bias:  if True, use bias
            device:  device
            dropout_wts:  dropout rate
        """
        
        super(AWD_LSTM, self).__init__()
        
        self.input_size = input_size
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.device = device
        self.embedding_size = embedding_size

        # Embedding
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Layers
        self.layers = []
        for i in range(nlayers):
            if i == 0:
                cell_input_s = embedding_size
            else:
                cell_input_s = hidden_size

            if i == nlayers - 1:
                cell_output_s = embedding_size
            else:
                cell_output_s = hidden_size

            layer = nn.LSTMCell(cell_input_s, cell_output_s, bias=bias)
            layer = WeightDropout(layer, dropout_wts) # Weight dropping
            self.layers.append(layer)

        # Decoder
        self.decoder = nn.Linear(embedding_size, self.input_size)
        
        self.output = None

    def forward(self, x, hiddens):
        """
        Forward pass.

        Parameters:
            x: input (batched)
            h: hidden state vector from previous activation (previous hidden state vector and cell state vector).
               Dimension: ((1, 1, input_size), (1, 1, input_size))

        Returns: hidden state vector and cell state vector as a tuple
        """
        
        h, c = hiddens
        output = Tensor().to(self.device)
        x = self.embedding(x)
        print(x)
        
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
        decoded = self.decoder(reshaped) # (batch_size, input_size)
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1)) # (batch_size, input_size, input_size)

        return decoded, (h, c)

