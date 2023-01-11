import numpy as np
import torch
from torch import Tensor
import torch.utils.data as Data
import utils.csv_func as csv
import os


def load_csv_data(data_dir:str, label_path:str) -> np.ndarray:
    labels = csv.load(label_path, 'id')
    x_list = []
    y_list = []
    for index, row in labels.iterrows():
        df_path = os.path.join(data_dir, f'{index}.csv')
        df = csv.load(df_path, 'date')
        x = np.array(df, dtype=int)
        x = x.reshape(-1)
        y = row[:]
        x_list.append(x)
        y_list.append(y)
    # transfer Dataframe to array
    x_data = np.array(x_list)
    y_data = np.array(y_list)
    return x_data, y_data


def buil_dataloader(x_set:Tensor, y_set:Tensor, batch_size:int, seed:int):
    dataset = Data.TensorDataset(x_set, y_set)
    # split dataset
    size = len(dataset)
    train_size, val_size = round(0.8 * size), round(0.2 * size)
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = Data.random_split(dataset, [train_size, val_size], generator)
    # data_loader
    train_loader = Data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2)
    val_loader = Data.DataLoader(val_dataset,batch_size=len(val_dataset), shuffle=True,num_workers=2)
    return train_loader, val_loader
