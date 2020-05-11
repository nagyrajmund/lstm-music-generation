import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor 

# A simplified version of AWD-LSTM implment from scratch, code adapted from https://github.com/a-martyn/awd-lstm 
# TODO if this works, modify notebook version
class LSTMCell(nn.Module):
    """
    A single LSTM unit. May be stacked.
    """

    def __init__(self, input_size, output_size, bias=True):
        """
        Initialize cell.

        Parameters:
            input_size: input size
            output_size: output size
            bias: if True, use bias
        """

        super(LSTMCell, self).__init__()
        
        # Contains all weights for the 4 linear mappings of the input x
        # e.g. Wi, Wf, Wo, Wc
        self.i2h = nn.Linear(input_size, 4*output_size, bias=bias)
        # Contains all weights for the 4 linear mappings of the hidden state h
        # e.g. Ui, Uf, Uo, Uc
        self.h2h = nn.Linear(output_size, 4 * output_size, bias=bias)
        self.output_size = output_size

    def forward(self, x, hidden):
        """
        Forward pass.

        Parameters:
            x: input vector
            h: hidden state vector from previous activation

        Returns: hidden state vector and cell state vector
        """

        # unpack tuple (recurrent activations, recurrent cell state)
        h, c = hidden

        # Linear mappings : all four in one vectorised computation
        preact = self.i2h(x) + self.h2h(h)

        # Activations
        i = torch.sigmoid(preact[:, :self.output_size])                      # input gate
        f = torch.sigmoid(preact[:, self.output_size:2*self.output_size])    # forget gate
        g = torch.tanh(preact[:, 3*self.output_size:])                       # cell gate
        o = torch.sigmoid(preact[:, 2*self.output_size:3*self.output_size])  # ouput gate

        # Cell state computations: 

        c_t = torch.mul(f, c) + torch.mul(i, g)
        h_t = torch.mul(o, torch.tanh(c_t))

        return h_t, c_t

class WeightDropout(nn.Module):
    
    def __init__(self, module: LSTMCell, weight_p: float):
        super().__init__()
        self.module, self.weight_p = module, weight_p
            
        w = getattr(self.module.h2h, 'weight')
        self.register_parameter('weight_raw', nn.Parameter(w.data))
        self.module.h2h._parameters['weight'] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        
        "Apply dropout to the raw weights."
        raw_w = getattr(self, 'weight_raw')
        self.module.h2h._parameters['weight'] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self):
        self._setweights()
        return self.module.forward()

class AWD_LSTM(nn.Module):
    """
    AWD_LSTM that uses the following optimization strategies: temporal activation regularization, weight dropping,
    variable length backpropagation sequences. Additional methods may be employed from the outside, e.g. ASGD.
    """
    # TODO other dropout methods? e.g. variational dropout
    # TODO implement windowed bptt

    def __init__(self, ntokens: int, input_size: int, embedding_size: int, hidden_size: int, nlayers: int=3, 
                 bias: bool=True, device: str='cpu', dropout_wts: float=0.5):
        
        super(AWD_LSTM, self).__init__()
        
        self.ntokens = ntokens
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

            layer = LSTMCell(cell_input_s, cell_output_s, bias=bias)
            layer = WeightDropout(layer, dropout_wts) # Weight dropping
            self.layers.append(layer)

        # Decoder
        self.decoder = nn.Linear(embedding_size, self.ntokens)
        
        self.output = None

    def embedding_dropout(self, embed, words, p=0.1):
        """
        Taken from original authors code.
        TODO: re-write and add test
        """
        if not self.training:
            masked_embed_weight = embed.weight
        elif not p:
            masked_embed_weight = embed.weight
        else:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - p).expand_as(embed.weight) / (1 - p)
            masked_embed_weight = mask * embed.weight
    
        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1
    
        X = F.embedding(words, masked_embed_weight,
                        padding_idx, embed.max_norm, embed.norm_type,
                        embed.scale_grad_by_freq, embed.sparse)
        return X

    def forward(self, x, hiddens):
        """
        Forward pass.

        Parameters:
            x: input
            hiddens: tuple; input from previous activation
        """

        x = self.embedding_dropout(self.embedding, x, p=self.dropout_emb)
        
        h, c = hiddens
        output = Tensor().to(self.device)
        
        # Propagate through layers for each timestep
        for t in range(x.size(0)):    
            
            inp = x[t,:,:]
            h_prev = inp
            h_new = []
            c_new = []

            for i in range(self.nlayers):
                h_curr, c_curr = self.layer[i](h_prev, (h[i], c[i]))
                hprev = h[i]
                h_new.append(h_curr)
                c_new.append(c_curr)

            h = h_new
            c = c_new
            
            # TAR - temporal activation regularization
            output = th.cat((output, h[-1].unsqueeze(0)))
            
        self.output = output.detach()
        
        # Translate embedding vectors to tokens
        # TODO rewrite this because the dimensions are bad
        reshaped = output.view(output.size(0) * output.size(1), output.size(2))
        decoded = self.decoder(reshaped)
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

        return decoded, (h, c)

