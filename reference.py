# %%
import geopandas as gpd
import pandas as pd
import os

# %%
PATH = 'D:\\Deutschland\\FUB\\master_thesis\\data\\Reference_data\\polygons'
OUTPUT_PATH = 'D:\\Deutschland\\FUB\\master_thesis\\data\\gee\\output'
INPUT_SHP = 'inpolysites.shp'
OUTPUT_SHP = 'buffered_wgs_inpolysites.shp'
REF_CSV = 'reference.csv'
LABEL_CSV = 'labels.csv'

# %%
def load_shp_file() -> gpd.GeoDataFrame:
    in_path = os.path.join(PATH, INPUT_SHP)
    # load shp file in python
    gdf = gpd.read_file(in_path)
    print(f'import file {in_path}')
    # remove useless columns
    keys = ['OBJEKTART', 'NUART', 'FBEZ', 'BETR', 'REVIER', 'DIST', 'ABT', 
            'RWET','BI', 'AI_FOLGE', 'BEST_BEZ', 'STICHTAG', 'LWET', 'FEVERFAHRE', 
            'TURNUS', 'BU_WLRT', 'LWET_TEXT', 'MASSNAHMEN', 'NWW_KAT', 'SHAPE_AREA', 
            'SHAPE_LEN', 'NHB_BEZ', 'WEFLKZ', 'GUID_ABT', 'layer', 'path']
    gdf.drop(columns=keys, inplace=True)
    # add uuid to each polygon
    gdf['id'] = gdf.index + 1
    return gdf

# %%
def export_shp_file(data_frame:gpd.GeoDataFrame) -> None:
    out_path = os.path.join(PATH, OUTPUT_SHP)
    gpd.GeoDataFrame.to_file(data_frame, out_path)
    print(f'export file {out_path}')

# %%
def buffer() -> None:
    # import shp file
    polygons = load_shp_file()
    # buffer
    polygons["geometry"] = gpd.GeoDataFrame.buffer(polygons, -10)
    print("Buffer -10 m")
    # reproject
    polygons = polygons.to_crs(epsg=4326)
    print("Reproject to EPSG:4326")
    # export shp file
    export_shp_file(polygons)

# %%
def load_csv_file(filename:str) -> pd.DataFrame:
    in_path = os.path.join(OUTPUT_PATH, filename)
    df = pd.read_csv(in_path, sep=',', header=0, index_col=False)
    print(f'import file {in_path}')
    return df

# %%
def export_csv_file(df:pd.DataFrame, filename:str) -> None:
    out_path = os.path.join(OUTPUT_PATH, filename)
    df.to_csv(out_path, index=False)
    print(f'export file {out_path}')

# %%
def export_reference_data() -> None:
    df = load_shp_file()
    cols = ['BST2_BA_1', 'BST2_BA_2', 'BST2_BA_3', 'BST2_BA_4', 'BST2_BA_5', 'BST2_BA_6', 'BST2_BA_7', 'BST2_BA_8', 
        'BST2_BAA_1', 'BST2_BAA_2', 'BST2_BAA_3', 'BST2_BAA_4', 'BST2_BAA_5', 'BST2_BAA_6', 'BST2_BAA_7', 'BST2_BAA_8', 
        'BST3_BA_1', 'BST3_BA_2', 'BST3_BA_3', 'BST3_BA_4', 'BST3_BA_5', 'BST3_BA_6', 'BST3_BA_7', 'BST3_BA_8', 
        'BST3_BAA_1', 'BST3_BAA_2', 'BST3_BAA_3', 'BST3_BAA_4', 'BST3_BAA_5', 'BST3_BAA_6', 'BST3_BAA_7', 'BST3_BAA_8',
        'geometry']
    df = df.drop(df[df['BST1_BA_1'] == 0].index)
    df = df.drop(df[df['BST2_BA_1'] != 0].index)
    df.drop(columns=cols, inplace=True)
    export_csv_file(df, REF_CSV)

# %%
def build_label() -> None:
    ref = load_csv_file(REF_CSV)
    # 110: Spruce
    # 710: Beech
    # 0: other coniferous
    # -1: other deciduous
    cols = ['Spruce', 'Beech', 'Coniferous', 'Deciduous', 'id']
    labels = []
    for index, row in ref.iterrows():
        label = pd.DataFrame(columns=cols, index=[0])
        label.fillna(value=0, inplace=True)
        label['id'] = row[-1]
        for i in range(8):
            if row[i] == 110:
                label['Spruce'] += row[i + 8] / 100
            elif row[i] == 710:
                label['Beech'] += row[i + 8] / 100
            elif row[i] >= 200 and row[i] <= 590:
                label['Coniferous'] += row[i + 8] / 100
            elif row[i] >=600 and row[i] != 710:
                label['Deciduous'] += row[i + 8] / 100
        labels.append(label)
    output = pd.concat(labels, ignore_index=True)
    export_csv_file(output, LABEL_CSV)



if __name__ == '__main__':
    export_reference_data()
    build_label()