import numpy as np
import torch
from torch import nn, optim, Tensor
import torch.utils.data as Data
from typing import Tuple
import os
import sys
sys.path.append('../')
import utils.csv as csv
from models.transformer import TransformerClassifier
import utils.validation as val
import utils.plot as plot


# file path
PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee\\output'
DATA_DIR = os.path.join(PATH, 'monthly_mean')
LABEL_CSV = '_label.csv'
TITLE = 'transformer_classification'
label_path = os.path.join(PATH, LABEL_CSV)

# general hyperparameters
BATCH_SIZE = 128
LR = 0.001
EPOCH = 5
SEED = 12345

# hyperparameters for Transformer model
num_bands = 10
seq_len = 25
num_classes = 5
d_model = 8
nhead = 4
num_layers = 1
dim_feedforward = 8


def numpy_to_tensor(x_data:np.ndarray, y_data:np.ndarray) -> Tuple[Tensor, Tensor]:
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    # reduce dimention of y_set from (n, 1) to (n, )
    y_set = y_set.view(-1)
    return x_set, y_set


def build_dataloader(x_set:Tensor, y_set:Tensor, batch_size:int, seed:int):
    # dataset = Data.TensorDataset(x_set, y_set)
    # # split dataset
    # size = len(dataset)
    # train_size, val_size = round(0.8 * size), round(0.2 * size)
    # generator = torch.Generator().manual_seed(seed)
    # train_dataset, val_dataset = Data.random_split(dataset, [train_size, val_size], generator)
    # manually split dataset
    x_train = x_set[:1105]
    y_train = y_set[:1105]
    x_val = x_set[1105:]
    y_val = y_set[1105:]
    train_dataset = Data.TensorDataset(x_train, y_train)
    val_dataset = Data.TensorDataset(x_val, y_val)
    # data_loader
    train_loader = Data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    val_loader = Data.DataLoader(val_dataset,batch_size=32, shuffle=True,num_workers=4)
    return train_loader, val_loader


def train(model:nn.Module, epoch:int) -> None:
    model.train()
    good_pred = 0
    total = 0
    losses = []
    for i, (inputs, labels) in enumerate(train_loader):
        # exchange dimension 0 and 1 of inputs depending on batch_first or not
        inputs:Tensor = inputs.transpose(0, 1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # recording training accuracy
        good_pred += val.true_pred_num(labels, outputs)
        total += labels.size(0)
        # record training loss
        losses.append(loss.item())
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # average train loss and accuracy for one epoch
    acc = good_pred / total
    train_loss = np.average(losses)
    # record loss and accuracy
    train_epoch_loss.append(train_loss)
    train_epoch_acc.append(acc)
    print('Epoch[{}/{}] | Train Loss: {:.4f} | Train Accuracy: {:.2f}% '
        .format(epoch+1, EPOCH, train_loss, acc * 100), end="")


def validate(model:nn.Module):
    model.eval()
    with torch.no_grad():
        good_pred = 0
        total = 0
        losses = []
        for (inputs, labels) in val_loader:
            # exchange dimension 0 and 1 of inputs depending on batch_first or not
            inputs:Tensor = inputs.transpose(0, 1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # prediction
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # recording validation accuracy
            good_pred += val.true_pred_num(labels, outputs)
            total += labels.size(0)
            # record validation loss
            losses.append(loss.item())
        # average train loss and accuracy for one epoch
        acc = good_pred / total
        val_loss = np.average(losses)
        # record loss and accuracy
        val_epoch_loss.append(val_loss)
        val_epoch_acc.append(acc)
    print('| Validation Loss: {:.4f} | Validation Accuracy: {:.2f}%'
        .format(val_loss, 100 * acc))



if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # dataset
    x_data, y_data = csv.to_numpy(DATA_DIR, label_path)
    x_set, y_set = numpy_to_tensor(x_data, y_data)
    train_loader, val_loader = build_dataloader(x_set, y_set, BATCH_SIZE, SEED)
    # model
    model = TransformerClassifier(num_bands, seq_len, num_classes, d_model, nhead, num_layers, dim_feedforward).to(device)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), LR)
    # train and validate model
    train_epoch_loss = []
    val_epoch_loss = []
    train_epoch_acc = []
    val_epoch_acc = []
    print("Start training")
    for epoch in range(EPOCH):
        train(model, epoch)
        validate(model)
    # visualize loss and accuracy during training and validation
    plot.draw(train_epoch_loss, val_epoch_loss, 'loss', TITLE)
    plot.draw(train_epoch_acc, val_epoch_acc, 'accuracy', TITLE)
    print('Plot result successfully')
    # save model
    # torch.save(model, '../outputs/model.pkl')