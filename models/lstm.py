import torch
from torch import nn,Tensor

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class LSTMClassifier(nn.Module):
    def __init__(self, num_bands:int, input_size:int, hidden_size:int, num_layers:int, num_classes:int, bidirectional:bool):
        super(LSTMClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 1
        if bidirectional:
            self.D = 2
        self.embd = nn.Linear(num_bands, input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Sequential(
                    nn.Linear(self.D * hidden_size, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )

    def forward(self, x:Tensor):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = self.embd(x)
        h0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:,-1,:])
        return out


class LSTMRegression(nn.Module):
    def __init__(self, num_bands:int, input_size:int, hidden_size:int, num_layers:int, num_classes:int, bidirectional:bool):
        super(LSTMRegression, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 1
        if bidirectional:
            self.D = 2
        self.embd = nn.Linear(num_bands, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Sequential(
                    nn.Linear(self.D * hidden_size, 256),
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
        h0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:,-1,:])
        return out


class LSTMMultiLabel(nn.Module):
    def __init__(self, num_bands:int, input_size:int, hidden_size:int, num_layers:int, num_classes:int, bidirectional:bool):
        super(LSTMMultiLabel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 1
        if bidirectional:
            self.D = 2
        self.embd = nn.Linear(num_bands, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Sequential(
                    nn.Linear(self.D * hidden_size, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                    nn.Sigmoid()
                )

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, num_classes)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = self.embd(x)
        h0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:,-1,:])
        return out