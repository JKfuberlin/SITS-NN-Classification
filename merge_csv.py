import os
import pandas as pd

PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee'
input_dir = os.path.join(PATH, 'extract')
output_dir = os.path.join(PATH, 'output')


def merge_csv_files(filename:str) -> None:
    # load all csv from folder
    # and turn into list of data frames
    input = []
    # add each csv file to input list as data frame
    files = os.listdir(input_dir)
    for file in files:
        if file.endswith(".csv"):
            in_path = os.path.join(input_dir, file)
            df = pd.read_csv(in_path, sep=',', header=0, index_col=False, parse_dates = ['date'])
            print(f'import file {in_path}')
            # delete date when no available data
            df = df.dropna(axis=0, how='any')
            # TODO: chooes same time series for each polygon
            df = reshape(df)
            input.append(df)
            # export each new csv file
            tmp_path = os.path.join(output_dir, file)
            df.to_csv(tmp_path, index=False)
            print(f"export reshaped file {tmp_path}")
    # merge all input data frames(runable)          
    output = pd.concat(input)
    out_path = os.path.join(output_dir, filename)
    output.to_csv(out_path, index=False)
    print(f"export merged file {out_path}")


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




if __name__ == "__main__":
    output_file = 'merged.csv'
    merge_csv_files(output_file)