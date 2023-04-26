import numpy as np
import torch
from torch import nn, Tensor
import torch.utils.data as Data
import os
import sys
import pandas as pd
import geopandas as gpd
sys.path.append('../')
import utils.csv as csv
import utils.shp as shp
import utils.plot as plot
from models.lstm import LSTMClassifier


# file path
PATH='/home/admin/dongshen/data'
DATA_DIR = os.path.join(PATH, 'gee', 'aoi_daily_padding')
LABEL_CSV = 'label_aoi.csv'
METHOD = 'classification'
MODEL = 'lstm'
UID = '8pure9'
MODEL_NAME = MODEL + '_' + UID
LABEL_PATH = os.path.join(PATH,'ref', 'validation', LABEL_CSV)
MODEL_PATH = f'../../outputs/models/{METHOD}/01/{MODEL_NAME}.pth'
SHP_PATH = os.path.join(PATH,'shp', 'aoi_polygons.shp')


# hyperparameters for LSTM
num_bands = 10
input_size = 64
hidden_size = 128
num_layers = 3
num_classes = 9
bidirectional = False


def build_dataloader(x_data:np.ndarray, y_data:np.ndarray) -> Data.DataLoader:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    # reduce dimention from (n, 1) to (n, )
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    # standardization
    sz, seq = x_set.size(0), x_set.size(1)
    x_set = x_set.view(-1, num_bands)
    batch_norm = nn.BatchNorm1d(num_bands)
    x_set:Tensor = batch_norm(x_set)
    x_set = x_set.view(sz, seq, num_bands).detach()
    # build dataset and dataloader
    dataset = Data.TensorDataset(x_set, y_set)
    dataloader = Data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    return dataloader


def predict(dataloader:Data.DataLoader, model:nn.Module) -> pd.DataFrame:
    """use neural network to make prediction"""
    model.eval()
    with torch.no_grad():
        y_list = []
        for (inputs, refs) in dataloader:
            inputs:Tensor = inputs.to(device)
            outputs:Tensor = model(inputs)
            # transfer prediction to class index
            _, predicted = torch.max(outputs.data, 1)
            refs[:, 1] = predicted
            # export as Dataframe
            y_list += refs.tolist()
    cols = ['id', 'class']
    pred = csv.list_to_dataframe(y_list, cols, False)
    csv.export(pred, f'../../outputs/csv/map/{METHOD}/{MODEL_NAME}_pred.csv', True)
    return pred


def map_class(pred:pd.DataFrame, gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """map class info to shp file"""
    # *****************change id column here*****************
    # gdf.rename(columns={'fid':'id'}, inplace=True)
    # *******************************************************
    output = pd.merge(gdf, pred, on='id', how='inner')
    class_name = {0:'Spruce', 1:'Silver Fir', 2:'Douglas Fir', 3:'Pine', 
                  4:'Oak', 5:'Red Oak', 6:'Beech', 7:'Sycamore', 8:'Others'}
    output['name'] = output['class'].map(class_name)
    return output


def validation_map(gdf:gpd.GeoDataFrame) -> None:
    """Draw validation map for 3 sub areas"""
    # Karlsruhe
    aoi_1 = gdf[gdf['Location'] == 'Hardtwald_pine_beech_redoak']
    # Stuttgart
    aoi_2 = gdf[gdf['Location'] == 'schoenbuch_beech_oak_mixture']
    # Freiburg
    aoi_3 = gdf[gdf['Location'] == 'Schwarzwald_spruce_silverfir_douglasfir']
    # dict of aoi
    areas = {'Hardtwald':aoi_1, 'Schoenbuch':aoi_2, 'Schwarzwald':aoi_3}
    # draw map
    for aoi, gdf in areas.items():
        plot.draw_map(gdf, aoi, MODEL_NAME)
    print('generate map successfully')



if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # dataset
    x_data, y_data = csv.to_numpy(DATA_DIR, LABEL_PATH)
    dataloader = build_dataloader(x_data, y_data)
    # model
    model = LSTMClassifier(num_bands, input_size, hidden_size, num_layers, num_classes, bidirectional).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    # make prediction
    print('start predicting')
    pred = predict(dataloader, model)
    gdf = shp.load_shp_file(SHP_PATH)
    # class mapping to shp
    output = map_class(pred, gdf)
    # drawing map
    validation_map(output)