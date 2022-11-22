import os
import pandas as pd
from typing import List


PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee'
INPUT_DIR = os.path.join(PATH, 'extract_cloud30')
OUTPUT_DIR = os.path.join(PATH, 'output')
DATE_FILE = 'occurrence_30.csv'
OUT_FILE = 'merged.csv'


def merge_csv_files(filename:str) -> None:
    # load all csv from folder
    # and turn into list of data frames
    input = []
    ref = load_reference_dates()
    # add each csv file to input list as data frame
    files = os.listdir(INPUT_DIR)
    for file in files:
        if file.endswith(".csv"):
            in_path = os.path.join(INPUT_DIR, file)
            df = pd.read_csv(in_path, sep=',', header=0, parse_dates = ['date'], index_col=['date'])
            print(f'import file {in_path}')
            df = shuffle(df, ref)
            df = monthly_mean_interpolate(df)
            # reshape columns to one row
            # df = reshape(df)
            # input.append(df)
            # export each new csv file
            tmp_path = os.path.join(OUTPUT_DIR, file)
            df.to_csv(tmp_path)
            print(f"export reshaped file {tmp_path}")
    # export_csv_file(input, filename)


def shuffle(df:pd.DataFrame, ref:pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame.join(ref, df)
    df.drop(columns=['count', 'spacecraft_id'],inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    return df


def monthly_mean_interpolate(df:pd.DataFrame) -> pd.DataFrame:
    # calculate mean band value in vegetation months for each year
    mean_df = df.resample('M').mean()
    mean_df = mean_df[((mean_df.index.month >= 5) & (mean_df.index.month <=9))]
    cols = list(mean_df.keys())
    for col in cols:
        mean_df[col] = mean_df[col].fillna(mean_df.groupby(mean_df.index.month)[col].transform('mean'))
    return mean_df


def reshape(df:pd.DataFrame) -> pd.DataFrame:
    # turn combination of reflectance value and date into a seperate column
    # new keys and values
    keys = list(df.keys())
    data = {'id':df.at[0, 'id']}
    for index, row in df.iterrows():
        date = row['date'].strftime('%Y%m%d')
        for key in keys[:-3]:
            column = f'{date} {key}'
            data[column] = [row[key]]
    # reshape data
    return pd.DataFrame(data)


def export_csv_file(data_frames:List[pd.DataFrame], filename:str) -> None:
    # merge all input data frames and export to csv file        
    output = pd.concat(data_frames)
    out_path = os.path.join(OUTPUT_DIR, filename)
    output.to_csv(out_path, index=False)
    print(f"export merged file {out_path}")


def load_reference_dates() -> pd.DataFrame:
    in_path = os.path.join(OUTPUT_DIR, DATE_FILE)
    df = pd.read_csv(in_path, sep=',', header=0, parse_dates = ['date'], index_col=['date'])
    return df



if __name__ == "__main__":
    merge_csv_files(OUT_FILE)