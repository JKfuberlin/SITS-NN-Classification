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
dropout=0.1
max_len = 5000 # defining maximum sequence length the model will be able to process here
# model dimensions (hyperparameter), dropout is already included here to make the model less prone to overfitting due to a specific sequence, max_length is defined. DOY needs to be smaller than that
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# I don't know if this makes sense here, as device should be assigned in the main script where model is trained / inference happens

class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence.
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
        pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model:int, dropout=0.1, max_len=5000): # model dimensions (hyperparameter), dropout is already included here to make the model less prone to overfitting due to a specific sequence, max_length is defined. DOY needs to be smaller than that
        super(PositionalEncoding, self).__init__() # The super() builtin returns a proxy object (temporary object of the superclass) that allows us to access methods of the base class.
        # i do not understand what that means
        self.dropout = nn.Dropout(p=dropout) # WTF i do not understand, what this does
        pe = torch.zeros(max_len, d_model) # positional encoding object is initialized with zeros, according to max length and model dimension. 5000 because we need a position on the sin/cos line for every possible DOY
        positionPE = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # used to create a 1-dimensional tensor representing the positions of tokens in a sequence.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # a tensor representing the values used for scaling in the positional encoding calculation
        # Apply the sinusoidal encoding to even indices and cosine encoding to odd indices
        pe[:, 0::2] = torch.sin(positionPE * div_term)
        pe[:, 1::2] = torch.cos(positionPE * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1) # torch.Size([5000, 1, 204]) max_len, ?, d_model
        self.register_buffer('pe', pe) #  Buffers are parameters that are not considered when computing gradients during backpropagation, we do not want to modify the PE

    def forward(self, doy):
        doy = doy.to(self.pe.device)
        return self.pe[doy, :]

    #     # x = x + self.pe[:x.size(0), :] # TODO: this is the problem, here something is added and it's not even DOY but something completely different, probably just the position in the sequence
    #     # the bigger issue is that is done in the wrong class.
    #     # TODO: I could just try to pass the pe object to the TransformerClassifier class and concat there
    #
    #     return self.dropout(x)
# TODO: Change this class to match the PE concatenation from pixel_based branch
class TransformerMultiLabel(nn.Module):
    def __init__(self, num_bands:int, num_classes:int, d_model:int, nhead:int, num_layers:int, dim_feedforward:int) -> None:
        super(TransformerMultiLabel, self).__init__()

        self.d_model = d_model
        # encoder embedding
        self.src_embd = nn.Linear(num_bands, d_model)
        # transformer model #duplicating d_model for PE concatenation in second half
        encoder_layer = TransformerEncoderLayer(d_model*2, nhead, dim_feedforward)
        encoder_norm = LayerNorm(d_model*2)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        output_size = d_model
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1) # global max pooling
        self.fc = nn.Sequential( # this takes the output of the transformer encoder after max pooling and passes it through linear layers to converge on num_classes
                    nn.Linear(output_size*2, 256),
                    nn.ReLU(),
                    nn.Linear(256, num_classes),
                )
        # regression

    def forward(self, input_sequence:Tensor, num_bands:int, num_classes:int) -> Tensor:
        if len(input_sequence.shape) == 2:
            # Add a batch dimension if it's not present (assuming batch size of 1)
            input_sequence = input_sequence.unsqueeze(1)
        input_sequence_bands = input_sequence[:,:,0:num_bands]  # this is the input sequence without DOY, shape should be [batch_size, seq_len, num_bands]
        # input_sequence_bands = input_sequence_bands.permute(0, 2, 1) # else use this to reshape
        obs_embed = self.src_embd(input_sequence_bands)  # [batch_size, seq_len, d_model] #
        self.PEinstance = PositionalEncoding(d_model=self.d_model, max_len=max_len)
        # this is where the input of form [batch_size, seq_len, n_bands] is passed through the linear transformation of the function src_embd()
        # to create the embeddings
        # Repeat obs_embed to match the shape [batch_size, seq_len, embedding_dim*2]
        x = obs_embed.repeat(1, 1, 2)
        # Add positional encoding based on day-of-year
        # X dimensions are [batch_size, seq_length, d_model*2], iterates over number of samples in each batch
        for i in range(input_sequence.size(0)):
            x[i, :, self.d_model:] = self.PEinstance(input_sequence[i, :, num_bands].long()).squeeze()
        # each batch's embedding is sliced and the second half replaced with a positional embedding of the DOY (11th column of the input_sequence) at the corresponding observation i
        output = self.transformer_encoder(x) # output: [seq_len, batch_size, d_model]
        output = self.global_max_pooling(output.permute(0,2,1)).squeeze(2)
        output = self.fc(output)  # GPT should be [batch_size, num_classes]
        # final shape: [batch_size, num_classes]
        return output