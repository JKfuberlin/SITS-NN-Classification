import torch
from torch import nn,Tensor
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMClassifier(nn.Module):
    def __init__(self, num_bands:int, input_size:int, hidden_size:int, num_layers:int, num_classes:int):
        super(LSTMClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embd = nn.Linear(num_bands, input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x:Tensor):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = self.embd(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        # tmp1 = self.relu(self.linear1(lstm_out[:,-1,:]))
        # tmp2 = self.relu(self.linear2(tmp1))
        out = self.fc(lstm_out[:,-1,:])
        return out


class LSTMRegression(nn.Module):
    def __init__(self, num_bands:int, input_size:int, hidden_size:int, num_layers:int, num_classes:int):
        super(LSTMRegression, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embd = nn.Linear(num_bands, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                    nn.Softmax(dim=1)
                )

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, num_classes)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = self.embd(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:,-1,:])
        return out