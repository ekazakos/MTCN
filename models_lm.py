# Model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MTCN_LM(nn.Module):
    def __init__(self,
                 num_class, 
                 d_model=512,  
                 dim_feedforward=512, 
                 nhead=8,
                 num_layers=4, 
                 dropout=0.1):
        super(MTCN_LM, self).__init__()
        self.num_class = num_class
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        self._create_model()

    def _create_model(self):
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=0.1)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model,
                                                           nhead=self.nhead,
                                                           dim_feedforward=self.dim_feedforward,
                                                           dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # (ntokens[0] + 1) and (ntokens[1] + 1) are MASK token for verb and noun respectively
        if isinstance(self.num_class, (list, tuple)):
            self.verb_encoder = nn.Embedding(self.num_class[0] + 1, self.d_model // 2)
            self.noun_encoder = nn.Embedding(self.num_class[1] + 1, self.d_model // 2)
            self.decoder = nn.Linear(self.d_model, self.num_class[0] + self.num_class[1])
        else:
            self.num_class = int(self.num_class)
            self.encoder = nn.Embedding(self.num_class + 1, self.d_model)
            self.decoder = nn.Linear(self.d_model, self.num_class)
        print("Building Transformer with {}-D, {} heads, and {} layers".format(self.d_model,
                                                                        self.nhead,
                                                                        self.num_layers))

    def forward(self, verb_input, noun_input=None):
        if isinstance(self.num_class, (list, tuple)):
            verb_src = self.verb_encoder(verb_input)
            noun_src = self.noun_encoder(noun_input)
            src = torch.cat([verb_src, noun_src], dim=-1)
        else:
            # For this option, the noun_input should be None
            assert noun_input == None
            src = self.encoder(verb_input)
            
        src *= math.sqrt(self.d_model)
        output = self.pos_encoder(src)
        output, _ = self.transformer_encoder(output)
        output = self.decoder(output)
        
        return output
