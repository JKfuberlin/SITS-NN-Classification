import numpy as np
import torch
from torch import nn, optim, Tensor
import torch.utils.data as Data
import os
import sys
import json
from typing import Tuple
sys.path.append('../')
import utils.csv as csv
from models.lstm import LSTMClassifier
import utils.validation as val
import utils.plot as plot
from datetime import datetime
import argparse
from sklearn.model_selection import train_test_split # for splitting the dataset in a stratified way

parser = argparse.ArgumentParser(description='trains LSTM given parameters')
parser.add_argument('UID', type=int, help='the unique ID of this particular model')
parser.add_argument('GPU_NUM', type=int, help='which GPU to use, necessary for parallelization')
parser.add_argument('input_size', type=int, help='input size')
parser.add_argument('hidden_size', type=int, help='hidden layer size')
parser.add_argument('num_layers', type=int, help='number of layers')
parser.add_argument('bidirectional', type=bool, help='True = bidirectional, False = normal onedirectional LSTM')

args = parser.parse_args()

# settings
PATH='/scratch/jonathak90/'
DATA_DIR = os.path.join(PATH, 'balanced_subset2')
LABEL_CSV = 'labels_clean.csv'
MODEL = 'lstm'
UID = args.UID
MODEL_NAME = MODEL + '_' + UID
LABEL_PATH = os.path.join(PATH, LABEL_CSV)
MODEL_PATH = '/home/jonathak90/outputs/models/'+MODEL_NAME+'.pth' #  TODO fix these joins

# general hyperparameters
BATCH_SIZE = 512
LR = 0.001
EPOCH = 200
SEED = 24

# fixed settings:
num_bands = 10 # number of columns in csv files

# hyperparameters for LSTM and argparse
input_size = args.input_size # larger
hidden_size = args.hidden_size # larger
num_layers = args.num_layers # larger
bidirectional = args.bidirectional


def save_hyperparameters() -> None:
    """Save hyperparameters into a json file"""
    params = {
        'general hyperparameters': {
            'batch size': BATCH_SIZE,
            'learning rate': LR,
            'epoch#': EPOCH,
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
    out_path = os.path.join('/home/jonathak90/outputs/models/',MODEL_NAME,'params.json')
    with open(out_path, 'w') as f:
        data = json.dumps(params, indent=4)
        f.write(data)
    print('saved hyperparameters')

def setup_seed(seed:int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def numpy_to_tensor(x_data:np.ndarray, y_data:np.ndarray) -> Tuple[Tensor, Tensor]:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    # reduce dimension from (n, 1) to (n, )
    y_data = y_data.reshape(-1)
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    # standardization
    sz, seq = x_set.size(0), x_set.size(1) # retrieve amount of samples and sequence length from tensor object
    x_set = x_set.view(-1, num_bands) # WTF does this do?
    batch_norm = nn.BatchNorm1d(num_bands)
    x_set: Tensor = batch_norm(x_set)
    x_set = x_set.view(sz, seq, num_bands).detach()
    return x_set, y_set

def build_dataloader(x_set:Tensor, y_set:Tensor, batch_size:int) -> Tuple[Data.DataLoader, Data.DataLoader]:
    """Build and split dataset, and generate dataloader for training and validation"""
    # automatically split dataset
    dataset = Data.TensorDataset(x_set, y_set) # what does this do? 'wrapping' tensors
    size = len(dataset)
    train_size, val_size = round(0.8 * size), round(0.2 * size)
    generator = torch.Generator() # this is for random sampling
    train_dataset, val_dataset = Data.random_split(dataset, [train_size, val_size], generator)
    # Create PyTorch data loaders from the datasets
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # num_workers is for parallelizing this function
    # shuffle is True so data will be shuffled in every epoch, this probably is activated to decrease overfitting
    # make sure, this does not mess up the proportions of labels
    return train_loader, val_loader

def train(model:nn.Module, epoch:int) -> Tuple[float, float]:
    model.train()
    good_pred = 0
    total = 0
    losses = []
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs[:,:, 1:11] # this excludes DOY and date
        # put the data in gpu
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
    print('Epoch[{}/{}] | Train Loss: {:.4f} | Train Accuracy: {:.2f}% '
        .format(epoch+1, EPOCH, train_loss, acc * 100), end="")
    return train_loss, acc

def validate(model:nn.Module) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        good_pred = 0
        total = 0
        losses = []
        for (inputs, labels) in val_loader:
            inputs = inputs[:, :, 1:11] # this excludes DOY and date
            # put the data in gpu
            inputs:Tensor = inputs.to(device)
            labels:Tensor = labels.to(device)
            # prediction
            outputs:Tensor = model(inputs)
            loss = criterion(outputs, labels)
            # recording validation accuracy
            good_pred += val.true_pred_num(labels, outputs)
            total += labels.size(0)
            # record validation loss
            losses.append(loss.item())
        # average train loss and accuracy for one epoch
        acc = good_pred / total
        val_loss = np.average(losses)
    print('| Validation Loss: {:.4f} | Validation Accuracy: {:.2f}%'
        .format(val_loss, 100 * acc))
    return val_loss, acc

def test(model:nn.Module, classes) -> None:
    """Test best model"""
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for (inputs, labels) in val_loader:
            inputs = inputs[:, :, 1:11] # this excludes DOY and date
            inputs:Tensor = inputs.to(device)
            labels:Tensor = labels.to(device)
            outputs:Tensor = model(inputs)  # again, this is where it somehow fails
            _, predicted = torch.max(outputs.data, 1)
            y_true += labels.tolist()
            y_pred += predicted.tolist()
        classes = range(classes)
        plot.draw_confusion_matrix(y_true, y_pred, classes, MODEL_NAME)

if __name__ == "__main__":
    # set random seed
    setup_seed(SEED)
    # Device configuration
    device = torch.device('cuda:'+args.GPU_NUM if torch.cuda.is_available() else 'cpu')
    # dataset
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    labels = csv.balance_labels_subset(LABEL_PATH, DATA_DIR)
    num_classes = 10  # number of different labels
    x_data, y_data = csv.to_numpy_subset(DATA_DIR, labels)
    x_set, y_set = numpy_to_tensor(x_data, y_data)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    train_loader, val_loader = build_dataloader(x_set, y_set, BATCH_SIZE)
    # model
    model = LSTMClassifier(num_bands, input_size, hidden_size, num_layers, num_classes, bidirectional).to(device)
    save_hyperparameters()
    # loss and optimizer
    # ******************change number of samples here******************
    # samples = torch.tensor([643, 254, 327, 1434])
    # mx = max(samples)
    # weight = torch.tensor([1., 1., 1., 1.1, 1.])
    # *****************************************************************
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), LR)
    # evaluate terms
    train_epoch_loss = []
    val_epoch_loss = []
    train_epoch_acc = [0]
    val_epoch_acc = [0]
    # train and validate model
    print("start training")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    for epoch in range(EPOCH):
        # print(epoch)
        train_loss, train_acc = train(model, epoch)
        val_loss, val_acc = validate(model)
        if val_acc > min(val_epoch_acc):
            torch.save(model.state_dict(), MODEL_PATH)
        # record loss and accuracy
        train_epoch_loss.append(train_loss)
        train_epoch_acc.append(train_acc)
        val_epoch_loss.append(val_loss)
        val_epoch_acc.append(val_acc)
    # visualize loss and accuracy during training and validation
    # plot.draw_curve(train_epoch_acc, val_epoch_acc, 'accuracy', MODEL_NAME)
    # test best model
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print('start testing')
    model.load_state_dict(torch.load(MODEL_PATH))
    # test(model)
    print('plot results successfully')
    torch.save(model, '/home/jonathak90/outputs/models/bi-lstm_demo.pkl')