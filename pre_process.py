import geopandas as gpd
import os


path = 'D:\\Deutschland\\FUB\\master_thesis\\data\\Reference_data\\polygons'


def load_shp_file(filename:str) -> gpd.GeoDataFrame:
    input_path = os.path.join(path, filename)
    # load shp file in python
    gdf = gpd.read_file(input_path)
    print(f'Imported file {filename} from {path}')
    # remove useless columns
    keys = ['OBJEKTART', 'NUART', 'FBEZ', 'BETR', 'REVIER', 'DIST', 'ABT', 
            'RWET','BI', 'AI_FOLGE', 'BEST_BEZ', 'STICHTAG', 'LWET', 'FEVERFAHRE', 
            'TURNUS', 'BU_WLRT', 'LWET_TEXT', 'MASSNAHMEN', 'NWW_KAT', 'SHAPE_AREA', 
            'SHAPE_LEN', 'NHB_BEZ', 'WEFLKZ', 'GUID_ABT', 'layer', 'path']
    gdf = gdf.drop(columns=keys)
    return gdf


def buffer(input_file:str, output_file:str) -> None:
    # import shp file
    polygons = load_shp_file(input_file)
    # add uuid to each polygon
    polygons['id'] = polygons.index + 1
    # buffer
    polygons["geometry"] = gpd.GeoDataFrame.buffer(polygons, -10)
    print("Buffer finished")
    # export shp file
    export_shp_file(polygons, output_file)
    

def export_shp_file(data_frame:gpd.GeoDataFrame, filename:str) -> None:
    output_path = os.path.join(path, filename)
    gpd.GeoDataFrame.to_file(data_frame, output_path)
    print(f'Exported file {filename} to {path}')



if __name__ == "__main__":
    input_file = 'inpolysites.shp'
    output_file = 'buffered_inpolysites.shp'
    buffer(input_file, output_file)