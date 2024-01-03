import pandas as pd
import numpy as np
from typing import Tuple, List
import os

def load(file_path:str, index_col:str, date=False) -> pd.DataFrame:
    """Load csv file to pandas.Dataframe"""
    # if date:
    #     df = pd.read_csv(file_path, sep=',', header=0, parse_dates = ['date'], index_col=index_col)
    #     # delete date when no available data
    #     df.dropna(axis=0, how='any', inplace=True)
    # else:
    df = pd.read_csv(file_path, sep=',', header=0, index_col=index_col)
    return df
def export(df:pd.DataFrame, file_path:str, index:bool) -> None:
    """Export pandas.Dataframe to csv file"""
    df.to_csv(file_path, index=index)
    print(f'export file {file_path}')
def delete(file_path:str) -> None:
    os.remove(file_path)
    print(f'delete file {file_path}')

def to_numpy(data_dir:str, label_path:str) -> Tuple[np.ndarray, np.ndarray]:
    """Load label and time series data, transfer them to numpy array"""
    print("load training data")
    labels = load(label_path, None)
    x_list = []
    y_list = []
    for index, row in labels.iterrows():
        df_path = os.path.join(data_dir, f'{row[0]}.csv')
        df = load(df_path, 'OBJECTID', False)
        x = np.array(df).astype(np.float32)
        y = row[:]
        x_list.append(x)
        y_list.append(y)
    # concatenate array list
    x_data = np.array(x_list)
    y_data = np.array(y_list)
    print("transfered data to numpy array")
    return x_data, y_data


def list_to_dataframe(lst:List[List[float]], cols:List[str], decimal:bool=True) -> pd.DataFrame:
    """Transfer list to pd.DataFrame"""
    df = pd.DataFrame(lst, columns=cols)
    df.set_index('id', inplace=True)
    if decimal:
        df = df.round(2)
    else:
        df = df.astype('int')
    return df