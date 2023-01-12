import numpy as np
import torch
from torch import nn, optim
from torchmetrics import R2Score
import os
import sys
sys.path.append('../')
import utils.dataset as dataset
from models.lstm import RegressionLSTM


# file path
PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee\\output'
DATA_DIR = os.path.join(PATH, 'monthly_mean')
LABEL_CSV = '5_classes.csv'
label_path = os.path.join(PATH, LABEL_CSV)

# general hyperparameters
BATCH_SIZE = 128
LR = 0.01
EPOCH = 100
SEED = 2048

# hyperparameters for LSTM
input_size = 32
hidden_size = 64
num_layers = 2
num_classes = 7


def build_dataset(x_data:np.ndarray, y_data:np.ndarray):
    # embedding
    embedding = nn.Embedding(8000, input_size)
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data).float()
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
        for (values, labels) in val_loader:
            values = values.to(device)
            labels = labels.to(device)
            outputs = model(values)
            num = labels.size(0)
            # transpose matrics
            outputs = outputs.t()
            labels = labels.t()
            r2score = R2Score(num_outputs=num, multioutput='raw_values').to(device)
        #     r2score = R2Score(num_outputs=num_classes, multioutput='raw_values').to(device)
            r2 = r2score(labels, outputs)
            good_pred = (r2 >= 0.8).sum().item()
        print(f'Portion of R^2 >= 0.8 on validate dataset: {good_pred / num * 100:.2f}%')
        # print(f'R^2 on validate set for each class:')
        # print('Spruce:{:.2f} | Beech:{:.2f} | Pine:{:.2f} | Douglas fir:{:.2f} | Oak:{:.2f} | Coniferous:{:.2f} | Deciduous:{:.2f}'
        #     .format(r2[0].item(), r2[1].item(), r2[2].item(), r2[3].item(), r2[4].item(), r2[5].item(), r2[6].item()))



if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset
    x_data, y_data = dataset.load_csv_data(DATA_DIR, label_path)
    x_set, y_set = build_dataset(x_data, y_data)
    train_loader, val_loader = dataset.build_dataloader(x_set, y_set, BATCH_SIZE, SEED)
    # model
    model = RegressionLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    # loss and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), LR)
    # train and validate model
    for epoch in range(EPOCH):
        train(model, epoch)
        validate(model)
    # save model
    # torch.save(model, '../outputs/model.pkl')


