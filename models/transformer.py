import torch
from torch import nn, Tensor
from torch.nn.modules.normalization import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


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

    def __init__(self, d_model:int, dropout=0.1, max_len=5000):
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


class TransformerRegression(nn.Module):
    def __init__(self, num_bands:int, num_classes:int, d_model:int, nhead:int, num_layers:int, dim_feedforward:int) -> None:
        super(TransformerRegression, self).__init__()
        # encoder embedding
        self.src_embd = nn.Linear(num_bands, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # transformer model
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # regression
        self.fc = nn.Sequential(
                    nn.Linear(d_model, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )

    def forward(self, src:Tensor) -> Tensor:
        src = self.src_embd(src)
        src = self.pos_encoder(src)
        output:Tensor = self.transformer_encoder(src)
        # output: [seq_len, batch_sz, d_model]
        output = self.fc(output[-1, :, :])
        # final shape: [batch_sz, num_classes]
        return output


class TransformerMultiLabel(nn.Module):
    def __init__(self, num_bands:int, num_classes:int, d_model:int, nhead:int, num_layers:int, dim_feedforward:int) -> None:
        super(TransformerMultiLabel, self).__init__()
        # encoder embedding
        self.src_embd = nn.Linear(num_bands, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # transformer model
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # regression
        self.fc = nn.Sequential(
                    nn.Linear(d_model, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                    nn.Sigmoid()
                )

    def forward(self, src:Tensor) -> Tensor:
        src = self.src_embd(src)
        src = self.pos_encoder(src)
        output:Tensor = self.transformer_encoder(src)
        # output: [seq_len, batch_sz, d_model]
        output = self.fc(output[-1, :, :])
        # final shape: [batch_sz, num_classes]
        return output


class TransformerClassifier(nn.Module):
    def __init__(self, num_bands:int, num_classes:int, d_model:int, nhead:int, num_layers:int, dim_feedforward:int) -> None:
        super(TransformerClassifier, self).__init__()
        # encoder embedding
        self.src_embd = nn.Linear(num_bands, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
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
                    nn.Linear(256, num_classes)
                )

    def forward(self, src:Tensor) -> Tensor:
        # src: [seq_len, batch_sz, num_bands]
        src = self.src_embd(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # output: [seq_len, batch_sz, d_model]
        output = self.fc(output[-1, :, :])
        # final shape: [batch_sz, num_classes]
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask