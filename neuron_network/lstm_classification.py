import numpy as np
import torch
from torch import nn, optim
import os
import sys
sys.path.append('../')
import utils.dataset as dataset
from models.lstm import ClassificationLSTM

# file path
PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee\\output'
DATA_DIR = os.path.join(PATH, 'monthly_mean')
LABEL_CSV = 'label.csv'
label_path = os.path.join(PATH, LABEL_CSV)

# general hyperparameters
BATCH_SIZE = 128
LR = 0.01
EPOCH = 100
SEED = 2048

# hyperparameters for LSTM
input_size = 5
hidden_size = 20
num_layers = 1
num_classes = 21
layer1_dim = 256
layer2_dim = 128


def build_dataset(x_data:np.ndarray, y_data:np.ndarray):
    # embedding
    embedding = nn.Embedding(8000, input_size)
    # reduce dimention from (n, 1) to (n, )
    y_data = y_data.reshape(-1)
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    x_set = embedding(x_set).detach()
    return x_set, y_set


def train(model:nn.Module, epoch:int):
    total_step = len(train_loader)
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record training loss
        if i % 40 == 0:
            print('Epoch[{}/{}],Step[{}/{}],Loss:{:.4f}'
            .format(epoch+1,EPOCH,i+40,total_step,loss.item()))


def validate(model:nn.Module):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for (values, labels) in val_loader:
            values = values.to(device)
            labels = labels.to(device)
            outputs = model(values)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model: {} %'.format(100 * correct / total))



if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset
    x_data, y_data = dataset.load_csv_data(DATA_DIR, label_path)
    x_set, y_set = build_dataset(x_data, y_data)
    train_loader, val_loader = dataset.build_dataloader(x_set, y_set, BATCH_SIZE, SEED)
    # model
    model = ClassificationLSTM(input_size, hidden_size, layer1_dim, layer2_dim, num_layers, num_classes).to(device)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), LR)
    # train and validate model
    for epoch in range(EPOCH):
        train(model, epoch)
        validate(model)
    # save model
    # torch.save(model, '../outputs/model.pkl')
