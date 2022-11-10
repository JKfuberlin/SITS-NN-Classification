import os
import pandas as pd

PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee'
input_dir = os.path.join(PATH, 'extract')
output_dir = os.path.join(PATH, 'output')

def merge_csv_files():
    # load all csv from folder
    # and turn into list of data frames
    input = []
    # add each csv file to input list as data frame
    for root,dirs,files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                dir = os.path.join(input_dir, file)
                df = pd.read_csv(dir, sep=',', header=0, index_col=False)
                input.append(df)
    # merge all input data frames           
    output = pd.concat(input)
    output_path = os.path.join(output_dir, 'merged.csv')
    output.to_csv(output_path, index=False)
    print("merged successfully")


# turn combination of reflectance value and date into a seperate column

# merge them



if __name__ == "__main__":
    merge_csv_files()