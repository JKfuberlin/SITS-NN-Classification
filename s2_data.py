# %%
import os
import pandas as pd
import utils.csv_func as csv

# %%
PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee'
INPUT_DIR = os.path.join(PATH,'extract_cloud30')
OUTPUT_DIR = os.path.join(PATH, 'output')
DATA_DIR = os.path.join(OUTPUT_DIR, 'monthly_mean')

DATE_CSV = 'occurrence_30.csv'
MERGE_CSV = 'merged.csv'
LABEL_CSV = 'labels.csv'

date_path = os.path.join(OUTPUT_DIR, DATE_CSV)
merge_path = os.path.join(OUTPUT_DIR, MERGE_CSV)

files = os.listdir(INPUT_DIR)

# %% [markdown]
# 1 Functions for Pandas DataFrame

# %%
def shuffle(df:pd.DataFrame, ref:pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(df, ref, how='right', on='date')
    df.drop(columns=['count', 'spacecraft_id', 'id'], inplace=True)
    # df.dropna(axis=0, how='any', inplace=True)
    return df

# %%
def monthly_mean_interpolate(df:pd.DataFrame) -> pd.DataFrame:
    # calculate mean band value in vegetation months for each year
    df.interpolate(method='time', inplace=True)
    mean_df = df.resample('M').mean()
    mean_df = mean_df[((mean_df.index.month >= 5) & (mean_df.index.month <=9))]
    # cols = list(mean_df.keys())
    # for col in cols:
    #     mean_df[col] = mean_df[col].fillna(mean_df.groupby(mean_df.index.month)[col].transform('mean'))
    return mean_df

# %%
def reshape(df:pd.DataFrame) -> pd.DataFrame:
    # turn combination of reflectance value and date into a seperate column
    # new keys and values
    keys = list(df.keys())
    data = {'id':int(df.iat[0, -1])}
    for index, row in df.iterrows():
        date = index.strftime('%Y%m%d')
        for key in keys[:-1]:
            column = f'{date} {key}'
            data[column] = [row[key]]
    # reshape data
    return pd.DataFrame(data)

# %% [markdown]
# 2 Count all available dates among all polygons

# %%
def count_dates() -> pd.DataFrame:
    map = {}
    # read each csv file
    for file in files:
        if file.endswith(".csv"):
            in_path = os.path.join(INPUT_DIR, file)
            try:
                df = csv.load(in_path, 'date', True)
                if df.empty:
                    csv.delete(in_path)
                    continue
                # count date occurrence
                for index, row in df.iterrows():
                    date = index.strftime('%Y%m%d')
                    map[date] = map.get(date, 0) + 1
            except Exception:
                # csv.delete(in_path)
                continue
    # export output as csv
    dates = list(map.keys())
    counts = list(map.values())
    output = pd.DataFrame({'date':dates, 'count':counts})
    output.sort_values(by='date', ascending=True, inplace=True)
    csv.export(output, date_path, index=False)
    return output

# %% [markdown]
# 3 Merge all data frames to one csv file

# %%
def merge_data_frame() -> pd.DataFrame:
    data_frames = []
    dates = csv.load(date_path, 'date', True)
    # add each csv file to input list as data frame
    for file in files:
        if file.endswith(".csv"):
            in_path = os.path.join(INPUT_DIR, file)
            df = csv.load(in_path, 'date', True)
            df = shuffle(df, dates)
            df = monthly_mean_interpolate(df)
            # reshape columns to one row
            # df = reshape(df)
            data_frames.append(df)
            # export each new csv file
            out_path = os.path.join(DATA_DIR, file[5:])
            csv.export(df, out_path, True)
        break
    # output = pd.concat(data_frames, ignore_index=True)
    # export_csv_file(output, merge_path, False)
    # return output

# %%
if __name__ == "__main__":
    count_dates()
    # merge_data_frame()


