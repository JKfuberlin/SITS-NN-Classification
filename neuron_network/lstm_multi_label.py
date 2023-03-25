import numpy as np
import torch
from torch import nn, optim, Tensor
import torch.utils.data as Data
from typing import Tuple
import os
import sys
import json
sys.path.append('../')
import utils.csv as csv
from models.lstm import LSTMMultiLabel
import utils.validation as val
import utils.plot as plot


# file path
PATH='/home/admin/dongshen/data'
DATA_DIR = os.path.join(PATH, 'gee', 'bw_8main_daily_padding')
LABEL_CSV = 'label_8multi.csv'
METHOD = 'multi_label'
MODEL = 'bi-lstm'
UID = '8ml'
MODEL_NAME = MODEL + '_' + UID
LABEL_PATH = os.path.join(PATH, 'ref', 'all',LABEL_CSV)
MODEL_PATH = f'../../outputs/models/{METHOD}/{MODEL_NAME}.pth'

# general hyperparameters
BATCH_SIZE = 512
LR = 0.001
EPOCH = 100
SEED = 24

# hyperparameters for LSTM
num_bands = 10
input_size = 64
hidden_size = 128
num_layers = 3
num_classes = 8
bidrectional = True


def setup_seed(seed:int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_hyperparameters() -> None:
    """Save hyperparameters into a json file"""
    params = {
        'general hyperparameters': {
            'batch size': BATCH_SIZE,
            'learning rate': LR,
            'epoch': EPOCH,
            'seed': SEED
        },
        f'{MODEL} hyperparameters': {
            'number of bands': num_bands,
            'embedding size': input_size,
            'hidden size': hidden_size,
            'number of layers': num_layers,
            'number of classes': num_classes
        }
    }
    out_path = f'../../outputs/models/{METHOD}/{MODEL_NAME}_params.json'
    with open(out_path, 'w') as f:
        data = json.dumps(params, indent=4)
        f.write(data)
    print('saved hyperparameters')


def numpy_to_tensor(x_data:np.ndarray, y_data:np.ndarray) -> Tuple[Tensor, Tensor]:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data).float()
    # standardization
    sz, seq = x_set.size(0), x_set.size(1)
    x_set = x_set.view(-1, num_bands)
    batch_norm = nn.BatchNorm1d(num_bands)
    x_set:Tensor = batch_norm(x_set)
    x_set = x_set.view(sz, seq, num_bands).detach()
    return x_set, y_set


def build_dataloader(x_set:Tensor, y_set:Tensor, batch_size:int) -> Tuple[Data.DataLoader, Data.DataLoader, Data.DataLoader]:
    """Build and split dataset, and generate dataloader for training and validation"""
    dataset = Data.TensorDataset(x_set, y_set)
    # split dataset
    size = len(dataset)
    train_size, val_size = round(0.8 * size), round(0.1 * size)
    test_size = size - train_size - val_size
    generator = torch.Generator()
    train_dataset, val_dataset, test_dataset = Data.random_split(dataset, [train_size, val_size, test_size], generator)
    # data_loader
    train_loader = Data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    val_loader = Data.DataLoader(val_dataset,batch_size=batch_size, shuffle=True,num_workers=4)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader
    

def train(model:nn.Module, epoch:int) -> Tuple[float, float]:
    model.train()
    accs = []
    losses = []
    for i, (inputs, refs) in enumerate(train_loader):
        labels:Tensor = refs[:,1:]
        # put the data in gpu
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # recording training accuracy
        accs.append(val.multi_label_acc(labels, outputs))
        # record training loss
        losses.append(loss.item())
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # average train loss and accuracy for one epoch
    acc = np.average(accs)
    train_loss = np.average(losses)
    print('Epoch[{}/{}] | Train Loss: {:.4f} | Train Accuracy: {:.2f}% '
        .format(epoch+1, EPOCH, train_loss, acc * 100), end="")
    return train_loss, acc


def validate(model:nn.Module) -> Tuple[float, float]:
    model.eval()
    accs = []
    losses = []
    with torch.no_grad():
        for (inputs, refs) in val_loader:
            labels:Tensor = refs[:,1:]
            # put the data in gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            # prediction
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # recording validation accuracy
            accs.append(val.multi_label_acc(labels, outputs))
            # record validation loss
            losses.append(loss.item())
        # average train loss and accuracy for one epoch
        acc = np.average(accs)
        val_loss = np.average(losses)
    print('| Validation Loss: {:.4f} | Validation Accuracy: {:.2f}%'
        .format(val_loss, 100 * acc))
    return val_loss, acc


def test(model:nn.Module) -> None:
    """Test best model"""
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for (inputs, refs) in test_loader:
            labels:Tensor = refs[:,1:]
            # put the data in gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs:Tensor = model(inputs)
            predicted = torch.where(outputs >= 0.5, 1, 0)
            y_true += refs.tolist()
            refs[:, 1:] = predicted
            y_pred += refs.tolist()
        # ***************************change classes here***************************
        cols = ['id','Spruce','Sliver Fir','Douglas Fir','Pine','Oak','Beech','Sycamore','Ash']
        # *************************************************************************
        ref = csv.list_to_dataframe(y_true, cols, False)
        pred = csv.list_to_dataframe(y_pred, cols, False)
        csv.export(ref, f'../../outputs/csv/{METHOD}/{MODEL_NAME}_ref.csv', True)
        csv.export(pred, f'../../outputs/csv/{METHOD}/{MODEL_NAME}_pred.csv', True)
        plot.draw_pie_chart(ref, pred, MODEL_NAME)
        plot.draw_multi_confusion_matirx(ref, pred, MODEL_NAME)



if __name__ == "__main__":
    # set random seed
    setup_seed(SEED)
    # Device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # dataset
    x_data, y_data = csv.to_numpy(DATA_DIR, LABEL_PATH)
    x_set, y_set = numpy_to_tensor(x_data, y_data)
    train_loader, val_loader, test_loader = build_dataloader(x_set, y_set, BATCH_SIZE)
    # model
    model = LSTMMultiLabel(num_bands, input_size, hidden_size, num_layers, num_classes, bidrectional).to(device)
    save_hyperparameters()
    # loss and optimizer
    # ******************change weight here******************
    # weight = torch.tensor([1., 1., 1., 1., 1.1, 1.1])
    # ******************************************************
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), LR)
    # evaluate terms
    train_epoch_loss = []
    val_epoch_loss = []
    train_epoch_acc = []
    val_epoch_acc = []
    max_val_acc = 0
    # train and validate model
    print("start training")
    for epoch in range(EPOCH):
        train_loss, train_acc = train(model, epoch)
        val_loss, val_acc = validate(model)
        if val_acc > max_val_acc:
            torch.save(model.state_dict(), MODEL_PATH)
            max_val_acc = val_acc
        # record loss and accuracy
        train_epoch_loss.append(train_loss)
        train_epoch_acc.append(train_acc)
        val_epoch_loss.append(val_loss)
        val_epoch_acc.append(val_acc)
    # visualize loss and accuracy during training and validation
    plot.draw_curve(train_epoch_loss, val_epoch_loss, 'loss', METHOD, MODEL_NAME)
    plot.draw_curve(train_epoch_acc, val_epoch_acc, 'accuracy', METHOD, MODEL_NAME)
    # draw scatter plot
    model.load_state_dict(torch.load(MODEL_PATH))
    test(model)
    print('plot result successfully')
