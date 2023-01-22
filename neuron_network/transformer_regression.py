import numpy as np
import torch
from torch import nn, optim, Tensor
import torch.utils.data as Data
from typing import Tuple
import os
import sys
sys.path.append('../')
import utils.csv as csv
from models.transformer import RegressionTransformer
import utils.validation as val


# file path
PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee\\output'
DATA_DIR = os.path.join(PATH, 'monthly_mean')
LABEL_CSV = '7_classes.csv'
label_path = os.path.join(PATH, LABEL_CSV)

# general hyperparameters
BATCH_SIZE = 128
LR = 0.01
EPOCH = 100
SEED = 12345

# hyperparameters for Transformer model
src_size = 8000
tgt_size = 101
d_model = 8
nhead = 4
num_layers = 1
dim_feedforward = 8


def numpy_to_tensor(x_data:np.ndarray, y_data:np.ndarray) -> Tuple[Tensor, Tensor]:
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    return x_set, y_set


def build_dataloader(x_set:Tensor, y_set:Tensor, batch_size:int, seed:int):
    dataset = Data.TensorDataset(x_set, y_set)
    # split dataset
    size = len(dataset)
    train_size, val_size = round(0.8 * size), round(0.2 * size)
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = Data.random_split(dataset, [train_size, val_size], generator)
    # # manually split dataset
    # x_train = x_set[:444]
    # y_train = y_set[:444]
    # x_val = x_set[444:]
    # y_val = y_set[444:]
    # train_dataset = Data.TensorDataset(x_train, y_train)
    # val_dataset = Data.TensorDataset(x_val, y_val)
    # data_loader
    train_loader = Data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2)
    val_loader = Data.DataLoader(val_dataset,batch_size=32, shuffle=True,num_workers=2)
    return train_loader, val_loader


def train(model:nn.Module, epoch:int) -> None:
    total_step = len(train_loader)
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.t().to(device)
        labels = labels.t().to(device)
        # forward pass
        outputs = model(inputs, labels)
        # labels = labels/100.
        # print(labels)
        loss = criterion(outputs, labels/100.)
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
    good_pred = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in val_loader:
            inputs = inputs.t().to(device)
            labels = labels.t().to(device)
            outputs = model(inputs, labels)
            good_pred += val.valid_num(labels/100., outputs)
            total += labels.size(1)
        print(f'Validation accuracy: {good_pred / total * 100:.2f}%')
        # print(f'R^2 on validate set for each class:')
        # print('Spruce:{:.2f} | Beech:{:.2f} | Pine:{:.2f} | Douglas fir:{:.2f} | Oak:{:.2f} | Coniferous:{:.2f} | Deciduous:{:.2f}'
        #     .format(r2[0].item(), r2[1].item(), r2[2].item(), r2[3].item(), r2[4].item(), r2[5].item(), r2[6].item()))



if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset
    x_data, y_data = csv.to_numpy(DATA_DIR, label_path)
    x_set, y_set = numpy_to_tensor(x_data, y_data)
    train_loader, val_loader = build_dataloader(x_set, y_set, BATCH_SIZE, SEED)
    # model
    model = RegressionTransformer(src_size, tgt_size, d_model, nhead, num_layers, dim_feedforward).to(device)
    # loss and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), LR)
    # train and validate model
    print("Start training")
    for epoch in range(EPOCH):
        train(model, epoch)
        validate(model)
        break
    # save model
    # torch.save(model, '../outputs/model.pkl')