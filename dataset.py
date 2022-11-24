# %%
import os
import pandas as pd
from typing import List

# %%
PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee'
INPUT_DIR = os.path.join(PATH, 'extract_cloud30')
OUTPUT_DIR = os.path.join(PATH, 'output')
DATE_CSV = 'occurrence_30.csv'
MERGE_CSV = 'merged.csv'

# %%
def load_csv_file(file_path:str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=',', header=0, parse_dates = ['date'], index_col=['date'])
    # delete date when no available data
    df.dropna(axis=0, how='any', inplace=True)
    print(f'import file {file_path}')
    return df

# %%
def export_csv_file(df:pd.DataFrame, file_path:str, index:bool) -> None:
    df.to_csv(file_path, index=index)
    print(f'export file {file_path}')

# %%
def delete_file(file_path:str) -> None:
    os.remove(file_path)
    print(f'delete file {file_path}')

# %%
def count_date() -> None:
    files = os.listdir(INPUT_DIR)
    map = {}
    # read each csv file
    for file in files:
        if file.endswith(".csv"):
            in_path = os.path.join(INPUT_DIR, file)
            try:
                df = load_csv_file(in_path)
                if df.empty:
                    delete_file(in_path)
                    continue
                # count date occurrence
                for index, row in df.iterrows():
                    date = index.strftime('%Y%m%d')
                    map[date] = map.get(date, 0) + 1
            except Exception:
                delete_file(in_path)
                continue
    # export output as csv
    dates = list(map.keys())
    counts = list(map.values())
    output = pd.DataFrame({'date':dates, 'count':counts})
    output.sort_values(by='date', ascending=True, inplace=True)
    out_path = os.path.join(OUTPUT_DIR, DATE_CSV)
    export_csv_file(output, out_path, index=False)

count_date()

# %%
def shuffle(df:pd.DataFrame, ref:pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame.join(ref, df)
    df.drop(columns=['count', 'spacecraft_id'],inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    return df

# %%
def monthly_mean_interpolate(df:pd.DataFrame) -> pd.DataFrame:
    # calculate mean band value in vegetation months for each year
    mean_df = df.resample('M').mean()
    mean_df = mean_df[((mean_df.index.month >= 5) & (mean_df.index.month <=9))]
    cols = list(mean_df.keys())
    for col in cols:
        mean_df[col] = mean_df[col].fillna(mean_df.groupby(mean_df.index.month)[col].transform('mean'))
    return mean_df

# %%
def reshape(df:pd.DataFrame) -> pd.DataFrame:
    # turn combination of reflectance value and date into a seperate column
    # new keys and values
    keys = list(df.keys())
    data = {'id':df.iat[0, -1]}
    for index, row in df.iterrows():
        date = index.strftime('%Y%m%d')
        for key in keys[:-1]:
            column = f'{date} {key}'
            data[column] = [row[key]]
    # reshape data
    return pd.DataFrame(data)

# %%
def merge_data_frame(data_frames:List[pd.DataFrame]) -> None:
    # merge all input data frames and export to csv file        
    merged_df = pd.concat(data_frames, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, MERGE_CSV)
    export_csv_file(merged_df, out_path, False)

# %%
def merge() -> None:
    # load all csv from folder
    # and turn into list of data frames
    data_frames = []
    date_path = os.path.join(OUTPUT_DIR, DATE_CSV)
    dates = load_csv_file(date_path)
    # add each csv file to input list as data frame
    files = os.listdir(INPUT_DIR)
    for file in files:
        if file.endswith(".csv"):
            in_path = os.path.join(INPUT_DIR, file)
            df =load_csv_file(in_path)
            df = shuffle(df, dates)
            df = monthly_mean_interpolate(df)
            # reshape columns to one row
            # df = reshape(df)
            data_frames.append(df)
            # export each new csv file
            out_path = os.path.join(OUTPUT_DIR, 'monthly_mean',file)
            export_csv_file(df, out_path, True)
    # merge_data_frame(data_frames)

merge()

# %%



