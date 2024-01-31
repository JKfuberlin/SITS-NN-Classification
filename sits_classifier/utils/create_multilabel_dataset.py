import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
import utils.csv as csv
import os

# # settings
PATH='/home/j/data/'
DATA_DIR = os.path.join(PATH, 'polygons_for_object_test_reshaped')
LABEL_CSV = 'multilabels.csv'
LABEL_PATH = os.path.join(PATH, LABEL_CSV)
# LABEL_PATH = '/home/jonathan/data/multilabels.csv'
# DATA_DIR = '/home/jonathan/data/multilabel_microdose/'

# year = 2022                     # single, recent year
# year = 2020:2022                # short, recent time series without drought year
# year = 2018:2022                # recent time series including drought years
# year = 2015:2017+2020:2022      # entire time series without drought years
# year = 2015:2022                # entire time series including drought years

if __name__ == "__main__":
    balance = False
    labels = csv.balance_labels_subset(LABEL_PATH, DATA_DIR, balance) # TODO: Revise if this step needs shuffling of labels
    x_data, y_data = csv.to_numpy(DATA_DIR, labels) # turn csv file into numpy dataset
    x_data.shape

    max_length = max(len(seq) for seq in x_data)     # Find the maximum length of sequences
    padded_sequences = [    # Pad sequences to the maximum length with nan values
        np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant', constant_values=np.nan)
        for seq in x_data
    ] # GPT, weird pad_width to avoid creation of new columns

    x_data = torch.tensor(padded_sequences)
    nan_mask = torch.isnan(x_data)
    # Replace NaN values with 0
    x_data[nan_mask] = 0

    a = x_data[0].shape
    num_bands = a[1]
    batch_norm = nn.BatchNorm1d(num_bands, eps=1e-3) # Create a BatchNorm1d layer with `num_bands` as the number of input features. # keepinmind: i am applying normalization to DOY, in theory should not affect positional Encoding but who knows...
    x_set = [
        batch_norm(seq)
        for seq in x_data
    ]

    # Convert the padded sequences to a PyTorch tensor
    y_set = torch.tensor(y_data)
    torch.save(x_set, '/home/j/data/x_set_mltlbl.pt')
    torch.save(y_set, '/home/j/data/y_set_mltlbl.pt')