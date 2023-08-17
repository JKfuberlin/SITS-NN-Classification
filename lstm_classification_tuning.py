import numpy as np
import torch
from torch import nn, optim, Tensor
import torch.utils.data as Data
import sys
import json
from typing import Tuple
sys.path.append('../')
from models.lstm import LSTMClassifier
import utils.validation as val
import utils.plot as plot
from utils.pytorchtools import EarlyStopping
from datetime import datetime
import argparse

local = True
parse = False

if parse == True:
    parser = argparse.ArgumentParser(description='trains LSTM given parameters')
    parser.add_argument('UID', type=int, help='the unique ID of this particular model')
    # parser.add_argument('GPU_NUM', type=int, help='which GPU to use, necessary for parallelization')
    parser.add_argument('input_size', type=int, help='input size')
    parser.add_argument('hidden_size', type=int, help='hidden layer size')
    parser.add_argument('num_layers', type=int, help='number of layers')
    parser.add_argument('batch_size', type=int, help='batch size')
    parser.add_argument('bidirectional', type=bool, help='True = bidirectional, False = normal onedirectional LSTM')
    args = parser.parse_args()
    # hyperparameters for LSTM and argparse
    input_size = args.input_size  # larger
    hidden_size = args.hidden_size  # larger
    num_layers = args.num_layers  # larger
    bidirectional = args.bidirectional
    BATCH_SIZE = args.batch_size
    UID = args.UID

print(f"UID = {UID}")

else:
    UID = 1
    input_size = 64 # according to GPT, this should not really be tuned but match the data structure
    hidden_size = 128
    num_layers = 3
    bidirectional = True
    BATCH_SIZE = 512

# general hyperparameters
LR = 0.001
EPOCH = 420
SEED = 420
patience = 1
# early stopping patience; how long to wait after last time validation loss improved.

if local == True:
    PATH = '/home/j/data/'
    MODEL = 'LSTM'
    MODEL_NAME = MODEL + '_' + str(UID)
    MODEL_PATH = '/home/j/data/outputs/models/' + MODEL_NAME   # TODO fix these joins
    EPOCH = 20
    x_set = torch.load('/home/j/data/x_set.pt')
    y_set = torch.load('/home/j/data/y_set.pt')


# fixed settings:
num_bands = 10 # number of columns in csv files
num_classes = 10  # number of different labels / different tree species classes in data

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
    out_path = MODEL_PATH+UID+'params.json'
    with open(out_path, 'w') as f:
        data = json.dumps(params, indent=4)
        f.write(data)
    print('saved hyperparameters')

def setup_seed(seed:int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True # https://darinabal.medium.com/deep-learning-reproducible-results-using-pytorch-42034da5ad7
    torch.backends.cudnn.benchmark = False # not sure if these lines are needed and non-deterministic algorithms would be used otherwise

def build_dataloader(x_set:Tensor, y_set:Tensor, batch_size:int) -> tuple[Data.DataLoader, Data.DataLoader]:
    """Build and split dataset, and generate dataloader for training and validation"""
    # automatically split dataset
    dataset = Data.TensorDataset(x_set, y_set) #  'wrapping' tensors: Each sample will be retrieved by indexing tensors along the first dimension.
    size = len(dataset)
    train_size, val_size = round(0.8 * size), round(0.2 * size)
    generator = torch.Generator() # this is for random sampling
    train_dataset, val_dataset = Data.random_split(dataset, [train_size, val_size], generator)
    # Create PyTorch data loaders from the datasets
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # num_workers is for parallelizing this function, however i need to set it to 1 on the HPC
    # shuffle is True so data will be shuffled in every epoch, this probably is activated to decrease overfitting
    # TODO: make sure, this does not mess up the proportions of classes seen by the training.
    return train_loader, val_loader

def train(model:nn.Module, epoch:int) -> tuple[float, float]:
    model.train()
    good_pred = 0
    total = 0
    losses = []
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs[:,:, 0:10] # this excludes DOY and date
        # print(inputs)
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

def validate(model:nn.Module) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        good_pred = 0
        total = 0
        losses = []
        for (inputs, labels) in val_loader:
            inputs = inputs[:, :, 0:10] # this excludes DOY and date
            inputs:Tensor = inputs.to(device)# put the data in gpu
            labels:Tensor = labels.to(device)
            outputs:Tensor = model(inputs) # prediction
            loss = criterion(outputs, labels)
            good_pred += val.true_pred_num(labels, outputs)# recording validation accuracy
            total += labels.size(0)
            losses.append(loss.item()) # record validation loss
        acc = good_pred / total  # average train loss and accuracy for one epoch
        val_loss = np.average(losses)
    print('| Validation Loss: {:.4f} | Validation Accuracy: {:.2f}%'
        .format(val_loss, 100 * acc))
    return val_loss, acc

# test() function is not used at the moment because i use the maps created by each model architecture and set of hyperparameters
# to determine accuracy using a completely independent validation dataset.
# However i might use some extra data from Betriebsinventur to compare models within this workflow in the future
# then i should take the test function from Dongshen's branch

def timestamp():
    now = datetime.now()
    current_time = now.strftime("%D:%H:%M:%S")
    print("Current Time =", current_time)

if __name__ == "__main__":
    setup_seed(SEED)  # set random seed to ensure reproducability
    # device = torch.device('cuda:'+args.GPU_NUM if torch.cuda.is_available() else 'cpu') # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device configuration
    timestamp()
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
    timestamp()
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
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
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping in epoch " + epoch)
            break
    # visualize loss and accuracy during training and validation
    model.load_state_dict(torch.load(MODEL_PATH))
    plot.draw_curve(train_epoch_loss, val_epoch_loss, 'loss',method='LSTM', model=MODEL_NAME)
    plot.draw_curve(train_epoch_acc, val_epoch_acc, 'accuracy',method='LSTM', model=MODEL_NAME)
    timestamp()
    # test(model)
    print('plot results successfully')
    torch.save(model, f'/home/j/data/outputs/models/{MODEL_NAME}.pkl')


# Annex 1 tensor.view() vs tensor.reshape()
#     view method:
#         The view method returns a new tensor that shares the same data with the original tensor but with a different shape.
#         If the new shape is compatible with the original shape (i.e., the number of elements remains the same), the view method can be used.
#         However, if the new shape is not compatible with the original shape (i.e., the number of elements changes), the view method will raise an error.
#
#     reshape method:
#         The reshape method also returns a new tensor with a different shape, but it may copy the data to a new memory location if necessary.
#         It allows reshaping the tensor even when the number of elements changes, as long as the new shape is compatible with the total number of elements in the tensor.