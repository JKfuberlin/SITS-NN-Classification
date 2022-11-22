import os
import pandas as pd

PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee'
INPUT_DIR = os.path.join(PATH, 'extract_cloud30')
OUTPUT_DIR = os.path.join(PATH, 'output')
OUT_FILE = 'occurrence_30.csv'


def count_date(out_file:str) -> None:
    files = os.listdir(INPUT_DIR)
    map = {}
    # read each csv file
    for file in files:
        if file.endswith(".csv"):
            in_path = os.path.join(INPUT_DIR, file)
            try:
                df = pd.read_csv(in_path, sep=',', header=0, parse_dates = ['date'], index_col=['date'])
                print(f'import file {in_path}')
                # delete date when no available data
                df = df.dropna(axis=0, how='any')
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
    output = output.sort_values(by='date', ascending=True)
    out_path = os.path.join(OUTPUT_DIR, out_file)
    output.to_csv(out_path, index=False)
    print(f'export file {out_path}')


def delete_file(path:str):
    os.remove(path)
    print(f'delete file {path}')



if __name__ == "__main__":
    count_date(OUT_FILE)