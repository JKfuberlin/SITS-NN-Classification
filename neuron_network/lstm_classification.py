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


# file path
PATH='D:\\Deutschland\\FUB\\master_thesis\\data'
DATA_DIR = os.path.join(PATH, 'gee', 'output', 'bw_pure_daily')
LABEL_CSV = 'label_7pure.csv'
METHOD = 'classification'
MODEL = 'lstm'
UID = '7pure'
MODEL_NAME = MODEL + '_' + UID
LABEL_PATH = os.path.join(PATH, 'ref', 'all', LABEL_CSV)
MODEL_PATH = f'../outputs/models/{METHOD}/{MODEL_NAME}.pth'

# general hyperparameters
BATCH_SIZE = 128
LR = 0.01
EPOCH = 5
SEED = 24

# hyperparameters for LSTM
num_bands = 10
input_size = 16
hidden_size = 16
num_layers = 1
num_classes = 7
bidirectional = False


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
    out_path = f'../outputs/models/{METHOD}/{MODEL_NAME}_params.json'
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
    # reduce dimention from (n, 1) to (n, )
    y_data = y_data.reshape(-1)
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    return x_set, y_set


def build_dataloader(x_set:Tensor, y_set:Tensor, batch_size:int) -> Tuple[Data.DataLoader, Data.DataLoader, Data.DataLoader]:
    """Build and split dataset, and generate dataloader for training and validation"""
    # # automatically split dataset
    # dataset = Data.TensorDataset(x_set, y_set)
    # size = len(dataset)
    # train_size, val_size = round(0.8 * size), round(0.2 * size)
    # generator = torch.Generator()
    # train_dataset, val_dataset = Data.random_split(dataset, [train_size, val_size], generator)
    # ------------------------------------------------------------------------------------------
    # manually split dataset
    # *******************change number here*******************
    x_train = x_set[:14212]
    y_train = y_set[:14212]
    x_val = x_set[14212: 15987]
    y_val = y_set[14212: 15987]
    x_test = x_set[15987:]
    y_test = y_set[15987:]
    # ******************************************************
    train_dataset = Data.TensorDataset(x_train, y_train)
    val_dataset = Data.TensorDataset(x_val, y_val)
    test_dataset = Data.TensorDataset(x_test, y_test)
    # data_loader
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


def train(model:nn.Module, epoch:int) -> Tuple[float, float]:
    model.train()
    good_pred = 0
    total = 0
    losses = []
    for i, (inputs, labels) in enumerate(train_loader):
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


def test(model:nn.Module) -> None:
    """Test best model"""
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for (inputs, labels) in test_loader:
            inputs:Tensor = inputs.to(device)
            labels:Tensor = labels.to(device)
            outputs:Tensor = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true += labels.tolist()
            y_pred += predicted.tolist()
        # *************************change class here*************************
        classes = ['Spruce','Douglas Fir','Pine','Oak','Red Oak','Beech','Sycamore']
        # *******************************************************************
        plot.draw_confusion_matrix(y_true, y_pred, classes, MODEL_NAME)



if __name__ == "__main__":
    # set random seed
    setup_seed(SEED)
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # dataset
    x_data, y_data = csv.to_numpy(DATA_DIR, LABEL_PATH)
    x_set, y_set = numpy_to_tensor(x_data, y_data)
    train_loader, val_loader, test_loader = build_dataloader(x_set, y_set, BATCH_SIZE)
    # model
    model = LSTMClassifier(num_bands, input_size, hidden_size, num_layers, num_classes, bidirectional).to(device)
    save_hyperparameters()
    # loss and optimizer
    # ******************change number of samples here******************
    samples = torch.tensor([24106/5, 3413, 1345, 2019, 1199, 8010/2, 964])
    weight = 1 / samples
    # *****************************************************************
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
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
    # test best model
    print('start testing')
    model.load_state_dict(torch.load(MODEL_PATH))
    test(model)
    print('plot results successfully')