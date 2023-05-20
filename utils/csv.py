import pandas as pd
import numpy as np
from typing import Tuple, List
import os


def load(file_path:str, index_col:str, date:bool=False) -> pd.DataFrame:
    """Load csv file to pandas.Dataframe"""
    if date:
        df = pd.read_csv(file_path, sep=',', header=0, parse_dates = ['date'], index_col=index_col)
        # delete date when no available data
        df.dropna(axis=0, how='any', inplace=True)
    else:
        df = pd.read_csv(file_path, sep=',', header=0, index_col=index_col)
    return df


def export(df:pd.DataFrame, file_path:str, index:bool=True) -> None:
    """Export pandas.Dataframe to csv file"""
    df.to_csv(file_path, index=index)
    print(f'export file {file_path}')


def delete(file_path:str) -> None:
    os.remove(file_path)
    print(f'delete file {file_path}')

def balance_labels(label_path:str):
    labels = load(label_path, 'ID')
    # find out least common class in labels
    # Count the occurrences of each label
    label_counts = labels["encoded"].value_counts()
    # Get the label with the least occurrences
    minority_label = label_counts.idxmin()
    # Get the number of occurrences of the minority label
    minority_count = label_counts[minority_label]

    dfs = []
    for label in label_counts.index:
        label_df = labels[labels["encoded"] == label]
        if len(label_df) > minority_count:
            label_df = label_df.sample(minority_count, random_state=42)
        dfs.append(label_df)
    # Concatenate the dataframes
    balanced_df = pd.concat(dfs)
    # Shuffle the dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=42)
    return balanced_df

def to_numpy(data_dir:str, labels) -> Tuple[np.ndarray, np.ndarray]:
    """Load label and time series data, transfer them to numpy array"""
    print("load training data")
    labels = labels # TODO i deleted the loading based on colname ID, make sure it works
    # Step 1: find max time steps
    max_len = 0
    for index, row in labels.iterrows():
        df_path = os.path.join(data_dir, f'{index}.csv') # TODO fix messed up file names
        df = load(df_path, 'date', True)
        max_len = max(max_len, df.shape[0])
    print(f'max sequence length: {max_len}')
    # Step 2: transfer to numpy array
    x_list = []
    y_list = []
    for index, row in labels.iterrows():
        df_path = os.path.join(data_dir, f'{index}.csv') # TODO: fix csv names
        df = load(df_path, 'date', True)
        x = np.array(df).astype(np.float32)
        # use 0 padding make sequence length equal
        padding = np.zeros((max_len - x.shape[0], x.shape[1]))
        x = np.concatenate((x, padding), dtype=np.float32)
        y = row['encoded']
        x_list.append(x)
        y_list.append(y)
    # concatenate array list
    x_data = np.array(x_list)
    y_data = np.array(y_list)
    print("transferred data to numpy array")
    return x_data, y_data


def list_to_dataframe(lst:List[List[float]], cols:List[str], decimal:bool=True) -> pd.DataFrame:
    """Transfer list to pd.DataFrame"""
    df = pd.DataFrame(lst, columns=cols)
    if decimal:
        df = df.round(2)
    else:
        df = df.astype('int')
    return df