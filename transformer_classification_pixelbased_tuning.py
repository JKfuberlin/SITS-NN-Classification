import numpy as np
import torch
from torch import nn, optim, Tensor
import torch.utils.data as Data
import sys
import json
from typing import Tuple
sys.path.append('../')
from models.transformer import TransformerClassifier
import utils.validation as val
import utils.plot as plot
from utils.pytorchtools import EarlyStopping
from datetime import datetime
import argparse
import os # for creating dirs if needed

local = True
parse = False

if parse == True:
    parser = argparse.ArgumentParser(description='trains the Transformer with given parameters')
    parser.add_argument('UID', type=int, help='the unique ID of this particular model')
    # parser.add_argument('GPU_NUM', type=int, help='which GPU to use, necessary for parallelization')
    parser.add_argument('d_model', type=int, help='d_model')
    parser.add_argument('nhead', type=int, help='number of transformer heads')
    parser.add_argument('num_layers', type=int, help='number of layers')
    parser.add_argument('dim_feedforward', type=int, help='')
    parser.add_argument('batch_size', type=int, help='batch size')
    args = parser.parse_args()
    # hyperparameters for LSTM and argparse
    d_model = args.d_model  # larger
    nhead = args.nhead  # larger
    num_layers = args.num_layers  # larger
    dim_feedforward = args.dim_feedforward
    BATCH_SIZE = args.batch_size
    UID = str(args.UID)
    print(f"UID = {UID}")

else:
    d_model = 204 # i want model dimension fit DOY_sequence length for now
    nhead = 4 # AssertionError: embed_dim must be divisible by num_heads
    num_layers = 2
    dim_feedforward = 512
    BATCH_SIZE = 32


# general hyperparameters
LR = 0.001 # learning rate, which in theory could be within the scope of parameter tuning
EPOCH = 420 # the maximum amount of epochs i want to train
SEED = 420 # a random seed for reproduction, at some point i should try different random seeds to exclude (un)lucky draws
patience = 25 # early stopping patience; how long to wait after last time validation loss improved.
num_bands = 11 # number of different bands
num_classes = 10 # the number of different classes that are supposed to be distinguished

if local == True:
    UID = 1
    PATH = '/home/j/data/'
    MODEL = 'Transformer'
    MODEL_NAME = MODEL + '_' + str(UID)
    MODEL_PATH = '/home/j/data/outputs/models/' + MODEL_NAME
    EPOCH = 20
    x_set = torch.load('/home/j/data/x_set2.pt')
    y_set = torch.load('/home/j/data/y_set2.pt')
    with open('/home/j/data/DOY_sequence.npy', 'rb') as f:
        DOY_sequence = np.load(f)

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
            'embedding size': d_model,
            'number of heads': nhead,
            'number of layers': num_layers,
            'feedforward dimension': dim_feedforward,
            'number of classes': num_classes
        }
    }
    out_path = f'../../outputs/models/{MODEL_NAME}_params.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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

def build_dataloader(x_set:Tensor, y_set:Tensor, batch_size:int, DOY_sequence:np.ndarray) -> tuple[Data.DataLoader, Data.DataLoader, Tensor]:
    """Build and split dataset, and generate dataloader for training and validation"""
    # automatically split dataset
    dataset = Data.TensorDataset(x_set, y_set) #  'wrapping' tensors: Each sample will be retrieved by indexing tensors along the first dimension.
    # x_set: [204, 305, 11] number of files, sequence length, number of bands
    size = len(dataset)
    train_size, val_size = round(0.8 * size), round(0.2 * size)
    generator = torch.Generator() # this is for random sampling
    # turn DOY sequence from np array into tensor
    doy_sequence_tensor = torch.from_numpy(DOY_sequence) # load the DOY sequence
    train_dataset, val_dataset = Data.random_split(dataset, [train_size, val_size], generator) # split the data in train and validation
    # Create PyTorch data loaders from the datasets
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1) # TODO: What is the DataLoader object? does it re-initialize every time it is called in train()?
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # num_workers is for parallelizing this function, however i need to set it to 1 on the HPC
    # shuffle is True so data will be shuffled in every epoch, this probably is activated to decrease overfitting
    # TODO: make sure, this does not mess up the proportions of classes seen by the training.
    return train_loader, val_loader, doy_sequence_tensor

def train(model:nn.Module, epoch:int, doy_sequence:Tensor) -> tuple[float, float]:
    doy_sequence = doy_sequence
    model.train()
    good_pred = 0
    total = 0
    losses = []
    for (inputs, labels) in train_loader:
        labels = labels.to(device)
        inputs = inputs.to(device) # pass the data into the gpu [3, 305, 11] 3?, sequence max length, num_bands
        inputs = torch.permute(inputs, (1, 0, 2))  # i need to switch the dimensions compared to LSTM to make the tensor match with the labels # from transformer.forward src: [seq_len, batch_sz, num_bands]
        outputs = model(inputs, DOY_sequence_tensor)  # applying the model
        # at this point inputs is 305,3,11. 305 [timesteps, ?, num_bands] TODO: Where does this come from, what is 2nd position and do i need all 11 bands? exclude DOY ?
        loss = criterion(outputs, labels)  # calculating loss by comparing to the y_set
        # recording training accuracy
        good_pred += val.true_pred_num(labels, outputs)
        total += labels.size(0)
        # record training loss
        losses.append(loss.item())
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = good_pred / total
    train_loss = np.average(losses)
    print('Epoch[{}/{}] | Train Loss: {:.4f} | Train Accuracy: {:.2f}% '.format(epoch + 1, EPOCH, train_loss, acc * 100), end="")
    return train_loss, acc


    for i, (inputs, labels) in enumerate(train_loader):
        # inputs = inputs[:,:, 0:10] # this was used for excluding date etc, as i do all the data preparation in create_dataset.py, i think it is no longer needed
        # pass the data into the gpu
        inputs = inputs.to(device)
        # from transformer.forward src: [seq_len, batch_sz, num_bands]
        inputs = torch.permute(inputs, (1,0,2)) # i need to switcheroo the dimensions compared to LSTM to make the tensor match with the labels
        labels = labels.to(device)
        # forward pass
        outputs = model(inputs, DOY_sequence_tensor) # applying the model
        loss = criterion(outputs, labels) # calculating loss by comparing to the y_set
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
            # inputs = inputs[:, :, 0:10] # this
            inputs:Tensor = inputs.to(device)# put the data in gpu
            inputs = torch.permute(inputs, (1, 0, 2))
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
    setup_seed(SEED)  # set random seed to ensure reproducibility
    # device = torch.device('cuda:'+args.GPU_NUM if torch.cuda.is_available() else 'cpu') # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device configuration
    timestamp()
    train_loader, val_loader, DOY_sequence_tensor = build_dataloader(x_set, y_set, BATCH_SIZE, DOY_sequence)
    # model
    model = TransformerClassifier(num_bands, num_classes, d_model, nhead, num_layers, dim_feedforward, DOY_sequence_tensor).to(device)
    save_hyperparameters()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), LR)
    softmax = nn.Softmax(dim=1).to(device)
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
        train_loss, train_acc = train(model, epoch, DOY_sequence_tensor)
        val_loss, val_acc = validate(model)
        if val_acc > min(val_epoch_acc):
            torch.save(model.state_dict(), MODEL_PATH)
            best_acc = val_acc
        # record loss and accuracy
        train_epoch_loss.append(train_loss)
        train_epoch_acc.append(train_acc)
        val_epoch_loss.append(val_loss)
        val_epoch_acc.append(val_acc)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping in epoch " + str(epoch))
            print("validation loss: " + str(best_acc))
            break
    # visualize loss and accuracy during training and validation
    model.load_state_dict(torch.load(MODEL_PATH))
    plot.draw_curve(train_epoch_loss, val_epoch_loss, 'loss',method='LSTM', model=MODEL_NAME, uid=UID)
    plot.draw_curve(train_epoch_acc, val_epoch_acc, 'accuracy',method='LSTM', model=MODEL_NAME, uid=UID)
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