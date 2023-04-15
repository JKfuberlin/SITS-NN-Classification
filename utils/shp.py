# %%
import geopandas as gpd
import utils.csv as csv


# %%
def load_shp_file(shp_path:str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    print(f'import file {shp_path}')
    return gdf

# %%
def export_shp_file(gdf:gpd.GeoDataFrame, shp_path:str) -> None:
    gpd.GeoDataFrame.to_file(gdf, shp_path)
    print(f'export file {shp_path}')

# %%
def shuffle(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # remove useless columns
    keys = ['OBJECTID', 'KATEGORIE', 'OBJEKTART', 'NUART', 'WEFLKZ', 'FBEZ', 'BETR', 'REVIER', 'DIST', 'ABT', 
    'RWET', 'BI', 'AI_FOLGE', 'BEST_BEZ', 'STICHTAG', 'LWET', 'ENTSTEHUNG', 'ENTSTEHU_1', 'AENDERUNG_', 
    'AENDERUNG1', 'LOCK__ID', 'GUID', 'GUID_ABT', 'GUID_DIS', 'GUID_BTR', 'GUID_REV', 'GUID_FBZ', 'GUID_NACHF', 
    'SFL_BEZ', 'GEFUEGE_BE', 'MASSN_BEZ', 'BRUCHBESTA', 'FEVERFAHRE', 'TURNUS', 'LWET_DARST', 'DAUERWALD', 
    'ALTKL_HB_A', 'ALTKL_IT_A', 'ALTKL_HB_B', 'BAA_PKT_TE', 'BAA_PKT_BU', 'BAA_PKT_EI', 'BAA_PKT_BL', 
    'BAA_PKT_PA', 'BAA_PKT_FI', 'BAA_PKT_TA', 'BAA_PKT_DG', 'BAA_PKT_KI', 'BAA_PKT_LA', 'BAA_PKT_NB', 
    'BAA_PKT_LB', 'GD_AB', 'GD_BIS', 'FATID', 'FOKUS_ID', 'GEFUEGE', 'BEST_BEZ1', 'BEST_BEZ2', 'BEST_BEZ3', 
    'BESTTYP', 'BU_WLRT', 'LWET_TEXT', 'MASSNAHMEN', 'NHB_BEZ', 'ALT_HB', 'ALT_IT', 'BST_ART_ID', 'NWW_BHT', 
    'NWW_KAT', 'SHAPE_STAr', 'SHAPE_STLe']
    gdf.drop(columns=keys, inplace=True)
    # remove useless rows
    gdf = gdf.drop(gdf[gdf['BST1_BA_1'] == 0].index)
    gdf = gdf.drop(gdf[gdf['BST2_BA_1'] != 0].index)
    gdf = gdf.drop(gdf[gdf['BST3_BA_1'] != 0].index)
    # add uuid to each polygon
    gdf['id'] = gdf.index + 1
    return gdf

# %%
def buffer(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # buffer -10 m
    gdf["geometry"] = gpd.GeoDataFrame.buffer(gdf, -10)
    print("Buffer -10 m")
    return gdf

# %%
def reproject(gdf:gpd.GeoDataFrame, epsg:int=4326) -> gpd.GeoDataFrame:
    # reproject to another coordinate system
    gdf = gdf.to_crs(epsg=epsg)
    print(f"Reproject to EPSG:{epsg}")
    return gdf

# %%
def select_pure(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Selcet pure classes whose main species is >= 90%"""
    gdf = gdf.drop(gdf[gdf['BST1_BAA_1'] < 90].index)
    return gdf

# %%
def select_8_main_classes(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """select polygons mainly contain 8 classes"""
    for i in range(1, 5):
        expr = f'(BST1_BA_{i}==0)|(BST1_BA_{i}==110)|(BST1_BA_{i}==210)|(BST1_BA_{i}==310)|(BST1_BA_{i}==410)|(BST1_BA_{i}==600)|(BST1_BA_{i}==710)|(BST1_BA_{i}==821)|(BST1_BA_{i}==831)'
        gdf.query(expr=expr, inplace=True)
    return gdf

# %%
def export_csv_reference(gdf:gpd.GeoDataFrame, ref_path:str) -> None:
    # delete unused columns
    cols = ['BST2_BA_1', 'BST2_BA_2', 'BST2_BA_3', 'BST2_BA_4', 'BST2_BA_5', 'BST2_BA_6', 'BST2_BA_7', 'BST2_BA_8', 
        'BST2_BAA_1', 'BST2_BAA_2', 'BST2_BAA_3', 'BST2_BAA_4', 'BST2_BAA_5', 'BST2_BAA_6', 'BST2_BAA_7', 'BST2_BAA_8', 
        'BST3_BA_1', 'BST3_BA_2', 'BST3_BA_3', 'BST3_BA_4', 'BST3_BA_5', 'BST3_BA_6', 'BST3_BA_7', 'BST3_BA_8', 
        'BST3_BAA_1', 'BST3_BAA_2', 'BST3_BAA_3', 'BST3_BAA_4', 'BST3_BAA_5', 'BST3_BAA_6', 'BST3_BAA_7', 'BST3_BAA_8',
        'geometry']
    gdf.drop(columns=cols, inplace=True)
    # export result as csv file
    csv.export(gdf, ref_path, False)