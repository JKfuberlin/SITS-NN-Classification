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
        self.register_buffer('pe', pe) # WTF i don't know what and why

    def forward(self, doy):
        return self.pe[doy, :]
    # Chris' solution

    # def forward_old(self, x):
    #     """forward function
    #
    #     applying dropout helps introduce some randomness or noise to the positional information.
    #     This randomness can be beneficial during training because it prevents the model from becoming
    #     overly reliant on specific positional patterns in the data.
    #
    #     x is the input sequence tensor of shape (sequence_length, batch_size, d_model) fed to the positional encoder model (required).
    #     output: [sequence length, batch size, embed dim]
    # self.pe[:x.size(0), :] extracts a subset of the precomputed positional encoding tensor (self.pe) based on the length of the input sequence.
    # This ensures that the positional encoding matches the length of the input sequence.
    # x = x + self.pe[:x.size(0), :] adds the positional encoding to the input sequence.
    # self.dropout(x) applies dropout to the resulting tensor. This is the dropout introduced during the initialization of the PositionalEncoding module.
    #     """
    #
    #     # x = x + self.pe[:x.size(0), :] # TODO: this is the problem, here something is added and it's not even DOY but something completely different, probably just the position in the sequence
    #     # the bigger issue is that is done in the wrong class.
    #     # TODO: I could just try to pass the pe object to the TransformerClassifier class and concat there
    #
    #     return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, num_bands:int, num_classes:int, d_model:int, nhead:int, num_layers:int, dim_feedforward:int) -> None:
        super(TransformerClassifier, self).__init__()

        self.d_model = d_model
        # encoder embedding, here a linear transformation is used to create the embedding, apparently it is an instance of nn.Linear that takes num_bands and d_model as args
        self.src_embd = nn.Linear(num_bands, d_model) # GPT: this linear transformation involves multiplying the input by a weight matrix and adding a bias vector.

        # transformer model
        encoder_layer = TransformerEncoderLayer(d_model*2, nhead, dim_feedforward)
        encoder_norm = LayerNorm(d_model*2)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # classification
        self.fc = nn.Sequential(
                    nn.Linear(d_model*2, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                    nn.Softmax(dim=1)
                )

    def forward(self, input_sequence: Tensor) -> Tensor:
        """
        Forward pass of the TransformerClassifier.

        Parameters:
            input_sequence (torch.Tensor): Input sequence tensor of shape (seq_len, batch_size, num_bands).
            doy_sequence (torch.Tensor): Day-of-year sequence tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        input_sequence_bands = input_sequence[:,:,0:10] # this is the input sequence without DOY
        obs_embed = self.src_embd(input_sequence_bands)  # [batch_size, seq_len, d_model] #
        self.PEinstance = PositionalEncoding(d_model=self.d_model, max_len=max_len)
        # this is where the input of form [batch_size, seq_len, n_bands] is passed through the linear transformation of the function src_embd()
        # to create the embeddings
        # Repeat obs_embed to match the shape [batch_size, seq_len, embedding_dim*2]
        x = obs_embed.repeat(1, 1, 2)
        # Add positional encoding based on day-of-year
        # X dimensions are [batch_size, seq_length, d_model*2], iterates over number of samples in each batch
        for i in range(input_sequence.size(0)):
            x[i, :, self.d_model:] = self.PEinstance(input_sequence[i, :, 10].long()).squeeze()
        #each batch's embedding is sliced and the second half replaced with a positional embedding of the DOY (11th column of the input_sequence) at the corresponding observation i
        output = self.transformer_encoder(x)
        # output: [seq_len, batch_size, d_model]
        output = output.mean(dim=1) #GPT WTF this is global max pooling, i am not sure what it means and how it works but it seemingly helps to attribute a single class to the entire time series instead of separate labels for each time step of the SITS
        output = self.fc(output) #GPT should be [batch_size, num_classes]
        # output = self.fc(output[-0, :, :]) # WTF is this subsetting??? i think i need either this or the global max pooling
        # final shape: [batch_size, num_classes]
        return output


        # src = self.src_embd(src)
        # src = self.pos_encoder(src)
        # output = self.transformer_encoder(src)
        # # output: [seq_len, batch_sz, d_model]
        # output = self.fc(output[-1, :, :])
        # # final shape: [batch_sz, num_classes]

# '''bugfix2'''
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model:int, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout) # WTF i do not understand, what this does
#         pe = torch.zeros(max_len, d_model) # positional encoding object is initialized with zeros, according to max length and model dimension. 5000 because we need a position on the sin/cos line for every possible DOY
#         positionPE = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # used to create a 1-dimensional tensor representing the positions of tokens in a sequence.
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # a tensor representing the values used for scaling in the positional encoding calculation
#         pe[:, 0::2] = torch.sin(positionPE * div_term)
#         pe[:, 1::2] = torch.cos(positionPE * div_term)
#         self.register_buffer('pe', pe) # WTF i don't know what and why
#     def forward(self, doy):
#         return self.pe[doy, :]
# PEinstance = PositionalEncoding(d_model=d_model, max_len=max_len)
# PEinstance
# input_sequence = inputs
# input_sequence_bands = input_sequence[:,:,0:10] # this is the input sequence without DOY
# src_embd = nn.Linear(num_bands, d_model)
# obs_embed = src_embd(input_sequence_bands)  # [batch_size, seq_len, d_model] # TODO: this should be 3,305,10 because DOY is not the object of embedding at thin point -> slice beforehand
# x = obs_embed.repeat(1, 1, 2)
#  # Add positional encoding based on day-of-year
# # X dimensions are [batch_size, seq_length, d_model*2], iterates over number of samples in each batch
#         for i in range(input_sequence.size(0)):
#             x[i, :, d_model:] = PEinstance(input_sequence[i, :, 10].long()).squeeze()
# #
# tobereplaced = x[i, :, d_model:]
# toreplace = PEinstance(input_sequence[i, :, 10].long()).squeeze()
# tobereplaced.size()
# toreplace.size()
#
# x.size() # torch.Size([15, 305, 1024])
# encoder_layer = TransformerEncoderLayer(d_model*2, nhead, dim_feedforward)
#         encoder_norm = LayerNorm(d_model*2)
#         transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
#         # classification
#         fc = nn.Sequential(
#                     nn.Linear(d_model*2, 256),
#                     nn.ReLU(),
#                     nn.BatchNorm1d(256),
#                     nn.Dropout(0.3),
#                     nn.Linear(256, num_classes),
#                     nn.Softmax(dim=1)
#                 )
#
#
# output = transformer_encoder(x)
# output.size() # gives torch.Size([15, 305, 1024])
# fc
# # output: [seq_len, batch_size, d_model]
# output2 = fc(output[-1, :, :])
# output2.size() # gives torch.Size([305, 10])
#
# '''chat GPT'''
# global_avg_pooled = output.mean(dim=1)  # Take the mean across the sequence dimension
#
# # Your final classification layer
# fc = nn.Sequential(
#     nn.Linear(d_model*2, 256),
#     nn.ReLU(),
#     nn.BatchNorm1d(256),
#     nn.Dropout(0.3),
#     nn.Linear(256, num_classes),
#     nn.Softmax(dim=1)
# )
#
# # Apply the classification layer to the global average pooled tensor
# output_final = fc(global_avg_pooled)




# '''local bugfixing'''
# max_len = 5000
# import math
# import torch
# from torch import nn, Tensor
# from torch.nn.modules.normalization import LayerNorm
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
#
# # init transformer
# src_embd = nn.Linear(num_bands-1, d_model) # create linear layer according to number of bands and wanted model dimension GPT: this linear transformation involves multiplying the input by a weight matrix and adding a bias vector.
# positionTC = PositionalEncoding(d_model=d_model, max_len=max_len) # do the positional encoding with the same model dimensions, fixed to the maximum length of a sequence
# encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward) # initialize an encoder layer object based on additional hyperparams
# encoder_norm = LayerNorm(d_model) # initialize normalization layer
# transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm) # i don't know what this does
# fc = nn.Sequential( # define a fully connected layer
#     nn.Linear(d_model, 256),
#     nn.ReLU(),
#     nn.BatchNorm1d(256),
#     nn.Dropout(0.3),
#     nn.Linear(256, num_classes),
#     nn.Softmax(dim=1) # TODO check if dim = 10 makes more sense for 10 classes -> input as sparse matrix probably necessary
# )
#
# input_sequence = inputs
# input_sequence_bands = inputs[:,:,0:9]
# doy_sequence = inputs[:,:,10]
# input_sequence2 = inputs # inputs is a 3/32, 305, 11 tensor [batch_size, sequence max length, num_bands]  aber laut Chris soll es sein [batch_size, seq_length, embedding_dim]->
# das wird es auch, wenn die naechste function ausgefuehrt wird

# run forward method
  # embed the input sequence [seq_len, batch_size, d_model]
# obs_embed = src_embd(input_sequence_bands)  #GPT: The input_sequence is the data you want to embed, and when you pass it through src_embd, it undergoes the linear transformation defined by the weight matrix and bias terms
# 3,305,11 -> 3,305,204 after embedding. This means, the 11 bands get 'translated' into d_model (204) because i define it as model dimension hyperparam.
# TODO: this should be 3,305,10 because DOY is not the object of embedding at this point -> slice beforehand
#
#
# x = obs_embed.repeat(1, 1, 2) # repeat the d_model??? i thought seq_len...
# #
# # what i want to do (i guess): add a pos encoding for each sequence
# # so one item consists of 11 bands, seq_len timesteps and seq_len DOY
# # # ntimeError: expand(torch.FloatTensor{[204, 1, 204]}, size=[204, 204]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (3)
# # x.size() is now torch.Size([3, 305, 408]), this means the d_model portion gets multiplicated by 2
# # doy_sequence = doy_sequence.long() # apparently i need to convert the DOY from float to long
# for i in range(input_sequence.size(0)): # X dimensions are [batch_size, seq_length, d_model*2], iterates over number of samples in each batch
#     x[i, :, d_model:] = positionTC(input_sequence[i, :, 10].long()).squeeze()  # i: takes each element of the batch and selects sequence and one half of the embedding vector to be replaced by
#     # positional encoding. The doy sequence is a [204, 305] - [d_model, seq_len] tensor
#
# c = x[0, :, d_model:]
# c.size()
# a = input_sequence[0, :, 9]
#
# d = positionTC(input_sequence[0, :, 9].long()).squeeze()
# d.size()()
# c = d
#
# toreplace = positionTC(input_sequence[i, :, 10].long())
# a = toreplace.squeeze()
# a.size()
# #
# b = input_sequence[i, :, 10].long()
# #
# tobereplaced = x[i, :, d_model:]
# tobereplaced.size()
#
# a=b
#
#     # debug for loop and see what it does
# for i in range(input_sequence.size(0)):
#     # print(x[i, :, d_model:])  # i think i try to replace the wrong dim of the tensor
#     print(position(doy_sequence[i,:]))
#
# tobereplaced = x[i, :, d_model:] # torch.Size([305, 204])
# toreplace = positionTC(doy_sequence[i, :]) # same error as above! This means that either position() is the problem or the internal structure and dimensions of doy_sequence
#
# doy_sequence.size()
# # Was macht eigentlich position()?
# # position() erwartet d_model, max_len, d.h. ich muss eventuell nur die dimensionsreihenfolge im Tensor tauschen oder auf die max_length padden
# # Versuch, dimensions tauschen:
# doy_sequence.size()
# doy_sequence2 = torch.permute(doy_sequence, (1, 0))
# doy_sequence2.size()
#
# for i in range(input_sequence.size(0)): # X dimensions are [batch_size, seq_length, d_model*2], iterates over number of samples in each batch
#     x[i, :, d_model:] = position(doy_sequence2[i, :])
# # RuntimeError: expand(torch.FloatTensor{[204, 1, 204]}, size=[305, 204]): the number of sizes provided (2)
# # must be greater or equal to the number of dimensions in the tensor (3)
# # Anderer Fehler als oben
# # TODO: Fact: der Fehler passiert nur in position(doy_sequence2[i, :])
# position = PositionalEncoding(d_model=d_model, max_len=max_len)
#
# # den ganzen shit debuggen:
# dropout=0.1
# max_len=5000
# # super(PositionalEncoding,self).__init__()  # The super() builtin returns a proxy object (temporary object of the superclass) that allows us to access methods of the base class.
# # i do not really understand why that is necessary
# dropout = nn.Dropout(p=dropout)  # i do not understand, what this does, maybe just initializing the dropout
# pe = torch.zeros(max_len,d_model)  # positional encoding object is initialized with zeros, according to max length and model dimension. this makes sense however 5000 seems excessive so removed a 0
# positionPE = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # used to create a 1-dimensional tensor representing the positions of tokens in a sequence.
# div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # a tensor representing the values used for scaling in the positional encoding calculation
# # Apply the sinusoidal encoding to even indices and cosine encoding to odd indices
# pe[:, 0::2] = torch.sin(positionPE * div_term)
# pe[:, 1::2] = torch.cos(positionPE * div_term)
# pe = pe.unsqueeze(0).transpose(0, 1) # torch.Size([5000, 1, 204]) max_len, ?, d_model
# # self.register_buffer('pe', pe)
# def forward(self, x):
#     x = x + self.pe[:x.size(0), :]
#     return self.dropout(x)
#
#
# position.size() # torch.Size([5000, 1])
# doy_sequence.size() # torch.Size([204, 305])
# toreplace = position(doy_sequence[i, :])
#
#
# tobereplaced.size()
# toreplace.size()
#
# print(len(doy_sequence[1, :]))
# print(len(doy_sequence[:, 0]))
# doy_sequence2 = torch.permute(doy_sequence, (1, 0))
#
#     # original by Chris:
# for i in range(batch_size):
#    x[i, :, self.embed_size:] = self.position(doy_sequence[i, :])  # [seq_length, embedding_dim]
#
# for i in range(x.size(1)): # x.size() = torch.Size([6, 305, 204]) x.size(1) = 305
#     x[:, i, src_embd.out_features:] = position(doy_sequence[:, i])
#
# # the number of sizes provided (2)
# whodis = src_embd.out_features
# x.size(1)
#
#
# # for i in range(x.size(1)):
# #     print((doy_sequence[:, i]))
# #
# # for i in range(batch_size):
# #         x[i, :, self.embed_size:] = self.position(doy_sequence[i, :])  # [seq_length, embedding_dim]
# #     x[i, :, embed_size:] = self.position(doy_sequence[i, :])  # [seq_length, embedding_dim]
#
#
#             # bugfixing
#             # for i in range(input_sequence.size(1)):
#             #     x[:, i, src_embd.out_features:] = position(doy_sequence[:, i])
#
#             output = transformer_encoder(x)
#             # output: [seq_len, batch_size, d_model]
#             output = fc(output[-1, :, :])
#             # final shape: [batch_size, num_classes]
#             return output