import os
import pandas as pd

PATH='D:\\Deutschland\\FUB\\master_thesis\\data\\gee'
input_dir = os.path.join(PATH, 'extract_date')
output_dir = os.path.join(PATH, 'output')


def count_date(out_file:str) -> None:
    files = os.listdir(input_dir)
    map = {}
    # read each csv file
    for file in files:
        if file.endswith(".csv"):
            in_path = os.path.join(input_dir, file)
            try:
                df = pd.read_csv(in_path, sep=',', header=0, index_col=False, parse_dates = ['date'])
                print(f'import file {in_path}')
                # delete date when no available data
                df = df.dropna(axis=0, how='any')
                # count date occurrence
                for index, row in df.iterrows():
                    date = row['date'].strftime('%Y%m%d')
                    map[date] = map.get(date, 0) + 1
            except Exception:
                continue
    # export output as csv
    dates = list(map.keys())
    counts = list(map.values())
    output = pd.DataFrame({'date':dates, 'count':counts})
    output = output.sort_values(by='date', ascending=True)
    out_path = os.path.join(output_dir, out_file)
    output.to_csv(out_path, index=False)
    print(f'export file {out_path}')



if __name__ == "__main__":
    out_file = 'occurrence.csv'
    count_date(out_file)