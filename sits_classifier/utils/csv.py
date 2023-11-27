import pandas as pd
import numpy as np
from typing import Tuple, List
import os

def custom_load(file_path:str, index_col:str, date:bool=False) -> pd.DataFrame:
    """Load csv file to pandas.Dataframe"""
    if date:
        df = pd.read_csv(file_path, sep=',', header=0, parse_dates = ['date'], index_col=False)
        # delete date when no available data
        df.dropna(axis=0, how='any', inplace=True)
    else:
        df = pd.read_csv(file_path, sep=',', header=0, index_col=False)
    return df

def export(df:pd.DataFrame, file_path:str, index:bool=True) -> None:
    """Export pandas.Dataframe to csv file"""
    df.to_csv(file_path, index=index)
    print(f'export file {file_path}')


def delete(file_path:str) -> None:
    os.remove(file_path)
    print(f'delete file {file_path}')

def balance_labels(label_path:str):
    labels = custom_load(label_path, 'ID')
    # find out the least common class in labels
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

def subset_filenames(data_dir:str):
    # i want to find out which csv files really are existent in my subset/on my drive and only select the matching labels
    import glob
    # Define the pattern to match the CSV files
    file_pattern = data_dir + '/*.csv'
    # Retrieve the filenames that match the pattern
    csv_files = glob.glob(file_pattern)
    # Extract the filenames without the extension
    file_names = [file.split('/')[-1].split('.')[0] for file in csv_files]
    file_names = [int(x) for x in file_names]
    return file_names

def balance_labels_subset(label_path:str, data_dir:str, balance:bool):
    file_names = subset_filenames(data_dir)
    labels = pd.read_csv(label_path, sep=',', header=0, index_col=False) # this loads all labels from the csv file
    try:
        labels = labels.drop("Unnamed: 0", axis=1) # just tidying up, removing an unnecessary column
        labels = labels.drop("X", axis=1)  # just tidying up, removing an unnecessary column
    except:
        print(labels)
    labels_subset = labels[labels['ID'].isin(file_names)] # drops all entries from the labels that do not have a corresponding csv file on the drive / in the subset
    # find out least common class in labels and count the occurrences of each label for balancing
    label_counts = labels_subset["encoded"].value_counts()
    minority_label = label_counts.idxmin()  # Get the label with the least occurrences
    minority_count = label_counts[minority_label] # Get the number of occurrences of the minority label
    labels = labels_subset # just resetting the variable name
    if balance == True:
        dfs = [] # empty list
        for label in label_counts.index:
            label_df = labels[labels["encoded"] == label]
            if len(label_df) > minority_count:
                label_df = label_df.sample(minority_count, random_state=42)
            dfs.append(label_df)
        # Concatenate the dataframes
        balanced_df = pd.concat(dfs)
    else:
        balanced_df = labels
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
        df = custom_load(df_path, 'date', True)
        max_len = max(max_len, df.shape[0])
    print(f'max sequence length: {max_len}')
    # Step 2: transfer to numpy array
    x_list = []
    y_list = []
    for index, row in labels.iterrows():
        df_path = os.path.join(data_dir, f'{index}.csv') # TODO: fix csv names
        df = custom_load(df_path, 'date', True)
        df = df.drop('date', axis=1) # i decided to drop the date again because i cannot convert it to float32 and i still have DOY for identification
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

def to_numpy_subset(data_dir:str, labels) -> Tuple[np.ndarray, np.ndarray]:
    """Load label and time series data, transfer them to numpy array"""
    labels = labels # at this point we already created cleaned up labels from a previous function
    # Step 1: find max time steps
    max_len = 0
    # TODO: something is messed up with the column names, i think naming one of them "ID" is a bad idea as it somehow gets switched around with the index
    for id in labels['ID']:
        df_path = os.path.join(data_dir, f'{id}.csv')
        df = custom_load(df_path, 'date', True)
        max_len = max(max_len, df.shape[0])
    print(f'max sequence length: {max_len}')
    # Step 2: transfer to numpy array
    x_list = []
    y_list = []

    for tuple in labels.iterrows():
        info = tuple[1] # access the first element of the tuple, which is a <class 'pandas.core.series.Series'>
        ID = info[0] # the true value for the ID after NA removal and some messing up is here, this value identifies the csv
        df_path = os.path.join(data_dir, f'{ID}.csv')
        df = custom_load(df_path, 'date', True)
        df = df.drop('date', axis=1)  # i decided to drop the date again because i cannot convert it to float32 and i still have DOY for identification
        x = np.array(df).astype(np.float32) # create a new numpy array from the loaded csv file containing spectral values with the dataype float32
        # use 0 padding make sequence length equal
        padding = np.zeros((max_len - x.shape[0], x.shape[1]))
        x = np.concatenate((x, padding), dtype=np.float32) # the 0s are appended to the end, will need to change this in the future to fill in missing observations
        y = info[2] # this is the label
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