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
def to_numpy(data_dir:str, labels:pd.core.frame.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load label and time series data, transfer them to numpy array"""
    print("load training data")
    labels = labels
    x_list = []
    y_list = []
    for index, row in labels.iterrows():
        df_path = os.path.join(data_dir, f'{row.iloc[0]}.csv')
        df = load(df_path, None, False)
        df = df.drop("date", axis = 1)
        x = np.array(df).astype(np.float32)
        y = row[:]
        x_list.append(x)
        y_list.append(y)
    # concatenate array list
    x_data = np.array(x_list, dtype=object)
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
    labels_subset = labels[labels['OBJECTID'].isin(file_names)] # drops all entries from the labels that do not have a corresponding csv file on the drive / in the subset
    # find out least common class in labels and count the occurrences of each label for balancing
    '''in case balancing is necessary, i need to think about redoing this'''
    # label_counts = labels_subset["label"].value_counts()
    # minority_label = label_counts.idxmin()  # Get the label with the least occurrences
    # minority_count = label_counts[minority_label] # Get the number of occurrences of the minority label
    # labels = labels_subset # just resetting the variable name
    # if balance == True:
    #     dfs = [] # empty list
    #     for label in label_counts.index:
    #         label_df = labels[labels["encoded"] == label]
    #         if len(label_df) > minority_count:
    #             label_df = label_df.sample(minority_count, random_state=42)
    #         dfs.append(label_df)
    #     # Concatenate the dataframes
    #     balanced_df = pd.concat(dfs)
    # else:
    balanced_df = labels
    # Shuffle the dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=42)
    return balanced_df
