import pandas as pd
import os


def load(file_path:str, index_col:str, date=False) -> pd.DataFrame:
    if date:
        df = pd.read_csv(file_path, sep=',', header=0, parse_dates = ['date'], index_col=index_col)
        # delete date when no available data
        df.dropna(axis=0, how='any', inplace=True)
    else: 
        df = pd.read_csv(file_path, sep=',', header=0, index_col=index_col)
    return df


def export(df:pd.DataFrame, file_path:str, index:bool) -> None:
    df.to_csv(file_path, index=index)
    print(f'export file {file_path}')


def delete(file_path:str) -> None:
    os.remove(file_path)
    print(f'delete file {file_path}')