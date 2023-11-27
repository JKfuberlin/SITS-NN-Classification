import torch
from torch import nn, Tensor
from torch.nn.modules.normalization import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

'''
This script defines an instance of the Transformer Object for Classification
It consists of two classes, one for Positional Encoding and another for the classifier itself
Both have a __init__ for initialization and a 'forward' method that is called during training and validation 
because they are both subclasses of nn.Module, and this is a required method in PyTorch's module system.

'''

max_len = 5000 # defining maximum sequence length the model will be able to process here
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# I don't know if this makes sense here, as device should be assigned in the main script where model is trained / inference happens

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
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


    def __init__(self, d_model:int, dropout=0.1, max_len=5000): # model dimensions (hyperparameter), dropout is already included here to make the model less prone to overfitting due to a specific sequence, max_length is defined. DOY needs to be smaller than that
        super(PositionalEncoding, self).__init__() # The super() builtin returns a proxy object (temporary object of the superclass) that allows us to access methods of the base class.
        # i do not understand what that means
        self.dropout = nn.Dropout(p=dropout) # i do not understand, what this does

        pe = torch.zeros(max_len, d_model) # positional encoding object is initialized with zeros, according to max length and model dimension. this makes sense however 5000 seems excessive so removed a 0
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # used to create a 1-dimensional tensor representing the positions of tokens in a sequence.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # a tensor representing the values used for scaling in the positional encoding calculation
        # Apply the sinusoidal encoding to even indices and cosine encoding to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """forward function

        applying dropout helps introduce some randomness or noise to the positional information.
        This randomness can be beneficial during training because it prevents the model from becoming
        overly reliant on specific positional patterns in the data.

        x is the input sequence tensor of shape (sequence_length, batch_size, d_model) fed to the positional encoder model (required).
        output: [sequence length, batch size, embed dim]
    self.pe[:x.size(0), :] extracts a subset of the precomputed positional encoding tensor (self.pe) based on the length of the input sequence.
    This ensures that the positional encoding matches the length of the input sequence.
    x = x + self.pe[:x.size(0), :] adds the positional encoding to the input sequence.
    self.dropout(x) applies dropout to the resulting tensor. This is the dropout introduced during the initialization of the PositionalEncoding module.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, num_bands:int, num_classes:int, d_model:int, nhead:int, num_layers:int, dim_feedforward:int, DOY_sequence_tensor) -> None:
        super(TransformerClassifier, self).__init__()
        doy_sequence = DOY_sequence_tensor

        # encoder embedding
        self.src_embd = nn.Linear(num_bands, d_model)

        # Positional encoding with sensitivity to distance between timesteps
        self.position = PositionalEncoding(d_model=d_model, max_len=max_len)

        # transformer model
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # classification
        self.fc = nn.Sequential(
                    nn.Linear(d_model, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                    nn.Softmax(dim=1)
                )

    def forward(self, input_sequence: Tensor, doy_sequence: Tensor) -> Tensor:
        """
        Forward pass of the TransformerClassifier.

        Parameters:
            input_sequence (torch.Tensor): Input sequence tensor of shape (seq_len, batch_size, num_bands).
            doy_sequence (torch.Tensor): Day-of-year sequence tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        obs_embed = self.src_embd(input_sequence)  # [seq_len, batch_size, d_model]
        # Repeat obs_embed to match the shape [seq_len, batch_size, embedding_dim*2]
        x = obs_embed.repeat(1, 1, 2)

        # Add positional encoding based on day-of-year
        for i in range(input_sequence.size(1)):
            x[:, i, self.src_embd.out_features:] = self.position(doy_sequence[:, i])

        output = self.transformer_encoder(x)
        # output: [seq_len, batch_size, d_model]
        output = self.fc(output[-1, :, :])
        # final shape: [batch_size, num_classes]
        return output

        # src = self.src_embd(src)
        # src = self.pos_encoder(src)
        # output = self.transformer_encoder(src)
        # # output: [seq_len, batch_sz, d_model]
        # output = self.fc(output[-1, :, :])
        # # final shape: [batch_sz, num_classes]


# '''local bugfixing'''
# max_len = 5000
# import math
#
# # init transformer
# doy_sequence = DOY_sequence_tensor
# src_embd = nn.Linear(num_bands, d_model)
# position = PositionalEncoding(d_model=d_model, max_len=max_len)
# encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
# encoder_norm = LayerNorm(d_model)
# transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
# fc = nn.Sequential(
#     nn.Linear(d_model, 256),
#     nn.ReLU(),
#     nn.BatchNorm1d(256),
#     nn.Dropout(0.3),
#     nn.Linear(256, num_classes),
#     nn.Softmax(dim=1)
# )
#
# input_sequence = inputs # what is my input_sequence???
# # run forward method
# obs_embed = src_embd(input_sequence)  # [seq_len, batch_size, d_model]
# x = obs_embed.repeat(1, 1, 2)
# #
# # # ntimeError: expand(torch.FloatTensor{[204, 1, 204]}, size=[204, 204]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (3)
# for i in range(input_sequence.size(1)):
#     x[:, i, src_embd.out_features:] = position(doy_sequence[:, i])
#
#
#             # bugfixing
#             # for i in range(input_sequence.size(1)):
#             #     x[:, i, src_embd.out_features:] = position(doy_sequence[:, i])
#
#             output = self.transformer_encoder(x)
#             # output: [seq_len, batch_size, d_model]
#             output = self.fc(output[-1, :, :])
#             # final shape: [batch_size, num_classes]
#             return output