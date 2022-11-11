import os
import pandas as pd

PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee'
input_dir = os.path.join(PATH, 'extract')
output_dir = os.path.join(PATH, 'output')


def merge_csv_files(filename:str):
    # load all csv from folder
    # and turn into list of data frames
    input = []
    # add each csv file to input list as data frame
    files = os.listdir(input_dir)
    for file in files:
        if file.endswith(".csv"):
            dir = os.path.join(input_dir, file)
            df = pd.read_csv(dir, sep=',', header=0, index_col=False, parse_dates = ['date'])
            # TODO: chooes same time series for each polygon
            df = reshape(df)
            input.append(df)
            # export each new csv file
            output_path = os.path.join(output_dir, file)
            df.to_csv(output_path, index=False)
            print(f"Exported reshaped {file} to {output_dir}")
    # # merge all input data frames(runable)          
    # output = pd.concat(input)
    # output_path = os.path.join(output_dir, filename)
    # output.to_csv(output_path, index=False)
    # print("merged successfully")


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