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
from models.transformer import TransformerMultiLabel
import utils.validation as val
import utils.plot as plot


LOCAL = True
PARSE = False
LOG = False
if LOG:
    logfile = '/tmp/logfile_transformer_pxl' # for logging model Accuracy
'''call http://localhost:6006/ for tensorboard to review profiling'''

if PARSE:
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
    d_model = 512 # i want model dimension fit DOY_sequence length for now
    # d_model = 128
    nhead = 4 # AssertionError: embed_dim must be divisible by num_heads
    num_layers = 6
    dim_feedforward = 256
    BATCH_SIZE = 16

if LOCAL:
    # file path
    PATH = '/home/j/data/'
    DATA_DIR = os.path.join(PATH, 'polygons_for_object_test_reshaped')
    LABEL_CSV = '/home/j/Nextcloud/csv/multilabels_test.csv'
    METHOD = 'multi_label'
    MODEL = 'transformer'
    UID = 'localtest'
    MODEL_NAME = MODEL + '_' + UID
    LABEL_PATH = os.path.join(PATH, 'ref', 'all', LABEL_CSV)
    MODEL_PATH = f'../../outputs/models/{METHOD}/{MODEL_NAME}.pth'

    x_set = torch.load('/home/j/data/x_set_mltlbl.pt')
    y_set = torch.load('/home/j/data/y_set_mltlbl.pt')
    d_model = 128 # i want model dimension fit DOY_sequence length for now
    # d_model = 128
    nhead = 1 # AssertionError: embed_dim must be divisible by num_heads
    num_layers = 1
    dim_feedforward = 64
    BATCH_SIZE = 8
    EPOCH = 20
    LR = 0.01  # learning rate, which in theory could be within the scope of parameter tuning
    if LOG:
        writer = SummaryWriter(log_dir='/home/j/data/prof/')  # initialize tensorboard
else:
    x_set = torch.load('/media/jonathan/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/x_set_pixelbased.pt')
    y_set = torch.load('/media/jonathan/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/y_set_pixelbased.pt')
    PATH = '/home/jonathan/data/'
    MODEL = 'Transformer'
    MODEL_NAME = MODEL + '_' + str(UID)
    MODEL_PATH = '/home/jonathan/data/outputs/models/' + MODEL_NAME
    EPOCH = 420  # the maximum amount of epochs i want to train
    LR = 0.00001  # learning rate, which in theory could be within the scope of parameter tuning
    if LOG:
        writer = SummaryWriter(log_dir='/home/jonathan/data/prof/')  # initialize tensorboard

# general hyperparameters
SEED = 420 # a random seed for reproduction, at some point i should try different random seeds to exclude (un)lucky draws
patience = 25 # early stopping patience; how long to wait after last time validation loss improved.
num_bands = 90 # number of different bands from Sentinel 2
num_classes = 12 # the number of different classes that are supposed to be distinguished

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
            'embedding size': d_model,
            'number of heads': nhead,
            'number of layers': num_layers,
            'feedforward dimension': dim_feedforward,
            'number of classes': num_classes
        }
    }
    out_path = f'../../outputs/models/{METHOD}/{MODEL_NAME}_params.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # create dir if necessary
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
    x_set_detached = x_set.detach()
    y_set_detached = y_set.detach()
    dataset = Data.TensorDataset(x_set_detached, y_set_detached)
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
import torch.nn.functional as F
def train(model:nn.Module, epoch:int) -> Tuple[float, float]:
    model.train()
    model.to(device)
    accs = []
    losses = []
    for i, (inputs, refs) in enumerate(train_loader):
        # exchange dimension 0 and 1 of inputs depending on batch_first or not
        inputs:Tensor = inputs.transpose(0, 1)
        labels:Tensor = refs[:,1:]
        inputs = inputs.permute(1, 0, 2) # I want the dimension order to be: batch, sequence, bands
        # put the data in gpu
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(inputs, num_bands, num_classes)
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1, labels.size(-1))
        # Use binary_cross_entropy_with_logits for each class separately
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())

        # loss = criterion(outputs, labels)
        # recording training accuracy
        outputs = sigmoid(outputs)
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
            # exchange dimension 0 and 1 of inputs depending on batch_first or not
            inputs:Tensor = inputs.transpose(0, 1)
            labels:Tensor = refs[:,1:]
            inputs = inputs.permute(1, 0, 2)  # I want the dimension order to be: batch, sequence, bands
            # put the data in gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            # prediction
            outputs = model(inputs, num_bands, num_classes)
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
            # recording validation accuracy
            outputs = sigmoid(outputs)
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
            # exchange dimension 0 and 1 of inputs depending on batch_first or not
            inputs:Tensor = inputs.transpose(0, 1)
            labels:Tensor = refs[:,1:]
            inputs = inputs.permute(1, 0, 2)
            # put the data in gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            # prediction
            outputs:Tensor = model(inputs, num_bands, num_classes)
            outputs = sigmoid(outputs)
            predicted = torch.where(outputs >= 0.5, 1, 0)
            y_true += refs.tolist()
            refs[:, 1:] = predicted
            y_pred += refs.tolist()
        # ***************************change classes here***************************
        cols = ['id','Spruce','Silver Fir','Douglas Fir','Pine','Oak','Beech','Sycamore','Sycamore','Sycamore','Sycamore','Sycamore','Sycamore',]
        # *************************************************************************
        ref = csv.list_to_dataframe(y_true, cols, False)
        pred = csv.list_to_dataframe(y_pred, cols, False)
        csv.export(ref, f'/home/j/data/outputs/1_ref.csv', True)
        csv.export(pred, f'/home/j/data/outputs/csv/1_pred.csv', True)
        plot.draw_pie_chart(ref, pred, MODEL_NAME)
        plot.draw_multi_confusion_matirx(ref, pred, MODEL_NAME)

if __name__ == "__main__":
    setup_seed(SEED)     # set random seed
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')     # Device configuration
    x_set = torch.stack(x_set, dim=0)
    train_loader, val_loader, test_loader = build_dataloader(x_set, y_set, BATCH_SIZE)
    # model
    model = TransformerMultiLabel(num_bands, num_classes, d_model, nhead, num_layers, dim_feedforward).to(device)
    save_hyperparameters()
    # loss and optimizer
    # ******************change weight here******************
    # num_positive = torch.tensor([28114, 19377, 8625, 6530, 2862, 19012, 2457], dtype=torch.float)
    # num_negative = torch.tensor([24362, 33099, 43851, 45946, 49614, 33464, 50019], dtype=torch.float)
    # pos_weight = num_negative / num_positive
    # ******************************************************
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), LR)
    sigmoid = nn.Sigmoid().to(device)
    # evaluate terms
    train_epoch_loss = []
    val_epoch_loss = []
    train_epoch_acc = []
    val_epoch_acc = []
    max_val_acc = 0
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
    