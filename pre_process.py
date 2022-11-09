import geopandas as gpd
import sys
import os

path = 'D:\\Deutschland\\FUB\\master_thesis\\data\\Reference_data\\polygons'
input_dir = os.path.join(path, sys.argv[1])
output_dir = os.path.join(path, sys.argv[2])

def buffer() -> None:
    polygons = gpd.read_file(input_dir)
    polygons["geometry"] = gpd.GeoDataFrame.buffer(polygons, -10)
    gpd.GeoDataFrame.to_file(polygons, output_dir)
    print("Buffer successfully!")



if __name__ == "__main__":
    buffer()