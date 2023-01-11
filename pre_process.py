import geopandas as gpd
import utils.csv_func as csv
import os


SHP_DIR = 'D:\\Deutschland\\FUB\\master_thesis\\data\\Reference_data\\polygons'
OUTPUT_DIR = 'D:\\Deutschland\\FUB\\master_thesis\\data\\gee\\output'
INPUT_SHP = 'inpolysites.shp'
OUTPUT_SHP = 'buffered_wgs_inpolysites.shp'
REF_CSV = 'reference.csv'


def load_shp_file() -> gpd.GeoDataFrame:
    in_path = os.path.join(SHP_DIR, INPUT_SHP)
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


def export_shp_file(data_frame:gpd.GeoDataFrame) -> None:
    out_path = os.path.join(SHP_DIR, OUTPUT_SHP)
    gpd.GeoDataFrame.to_file(data_frame, out_path)
    print(f'export file {out_path}')


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


def export_reference_data() -> None:
    df = load_shp_file()
    # delete unused columns
    cols = ['BST2_BA_1', 'BST2_BA_2', 'BST2_BA_3', 'BST2_BA_4', 'BST2_BA_5', 'BST2_BA_6', 'BST2_BA_7', 'BST2_BA_8', 
        'BST2_BAA_1', 'BST2_BAA_2', 'BST2_BAA_3', 'BST2_BAA_4', 'BST2_BAA_5', 'BST2_BAA_6', 'BST2_BAA_7', 'BST2_BAA_8', 
        'BST3_BA_1', 'BST3_BA_2', 'BST3_BA_3', 'BST3_BA_4', 'BST3_BA_5', 'BST3_BA_6', 'BST3_BA_7', 'BST3_BA_8', 
        'BST3_BAA_1', 'BST3_BAA_2', 'BST3_BAA_3', 'BST3_BAA_4', 'BST3_BAA_5', 'BST3_BAA_6', 'BST3_BAA_7', 'BST3_BAA_8',
        'geometry']
    df = df.drop(df[df['BST1_BA_1'] == 0].index)
    df = df.drop(df[df['BST2_BA_1'] != 0].index)
    df.drop(columns=cols, inplace=True)
    # export result as csv file
    ref_path = os.path.join(OUTPUT_DIR, REF_CSV)
    csv.export(df, ref_path, False)



if __name__ == "__main__":
    buffer()
    export_reference_data()