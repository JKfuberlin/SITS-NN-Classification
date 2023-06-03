import os # for general operating system functionality
import glob # for retrieving files using strings/regex
import re # for regex
import rasterio # for reading rasters
import geopandas as gpd # for reading shapefiles
import numpy as np
import datetime # for benchmarking
import torch # for loading the model and actual inference
import rioxarray as rxr # for raster clipping
import multiprocessing # for parallelization
from shapely.geometry import mapping # for clipping

tiles = np.loadtxt('/my_volume/BW_tiles.txt', dtype=str) # this file contains the XY tile names of my AOI in the same format as FORCE

device = torch.device('cpu') # assigning cpu for inference
model_pkl = torch.load('/my_volume/bi_lstm_demo.pkl',  map_location=torch.device('cpu')) # loading the trained model

def predict(input):
    print('.')
    outputs = model_pkl(input)
    _, predicted = torch.max(outputs.data, 1)
    print('*')
    return predicted

raster_paths = []

for tile in tiles:
    s2dir = glob.glob(('/force/FORCE/C1/L2/ard/'+tile+'/'), recursive=True) # Initialize path to Sentinel-2 time series
    raster_paths = [] # create object to be filled with rasters to be stacked
    tifs = glob.glob(os.path.join(s2dir[0], "*SEN*BOA.tif")) # i want to ignore the landsat files and stack only sentinel 2 bottom of atmosphere observations
    raster_paths.extend(tifs) # write the ones i need into my object to stack from
    years = [int(re.search(r"\d{8}", raster_path).group(0)) for raster_path in raster_paths] # i need to extract the date from each file path...
    raster_paths = [raster_path for raster_path, year in zip(raster_paths, years) if year <= 20230302] # ... to cut off at last image 20230302 as i trained my model with this sequence length, this might change in the future with transformers
    datacube = [rxr.open_rasterio(raster_path) for raster_path in raster_paths]  # i user rxr because the clip function from rasterio sucks
    # the datacube is too large for the RAM of my contingent so i need to subset using the 5x5km tiles
    grid_tiles = glob.glob(os.path.join('/my_volume/FORCE_tiles_subset_BW/', '*' + tile + '*.gpkg'))
    for minitile in grid_tiles: # now i iterate over the small grids
        crop_shape = gpd.read_file(minitile) # open the corresponding geometry
        clipped_datacube = [] # empty object for adding clipped rasters
        for raster in datacube: # clip the rasters using the geometry
            clipped_raster = raster.rio.clip(crop_shape.geometry.apply(mapping))
            clipped_datacube.append(clipped_raster)
    x=clipped_datacube.shape[2] # i need the dimensions of the raster to create an empty one
    y=clipped_datacube.shape[3]
    prediction = torch.zeros([x,y]) # create empty tensor where results will be written
    data_for_prediction = data_for_prediction.permute(2, 3, 0, 1) # rearrange data
    num_cores = 15 # Define the number of cores to use
    pool = multiprocessing.Pool(processes=num_cores) # Create a multiprocessing pool
    predictions = pool.map(predict, data_for_prediction) # Iterate over the range in parallel and store the results in a list
    # Close the pool to release resources
    pool.close()
    pool.join()
    predictions = torch.tensor(predictions) # Convert the list to an array
    map = prediction.numpy()
    with rasterio.open(raster_paths[0]) as src:  # i can use any image from the stack cuz they all the same
        metadata = src.meta
        # Update the number of bands in the metadata to match the modified image
        metadata['count'] = 1
    # writing
    with rasterio.open('/my_volume/map_BILSTM_'+tile+'_'+minitile, 'w',**metadata) as dst:  # open write connection to file using the metadata
        dst.write(map, 1)


# testing for a single tile

tile = tiles[14]
s2dir = glob.glob(('/force/FORCE/C1/L2/ard/' + tile + '/'), recursive=True)  # Initialize path to Sentinel-2 time series
raster_paths = []  # create object to be filled with rasters to be stacked
tifs = glob.glob(os.path.join(s2dir[0], "*SEN*BOA.tif"))  # i want to ignore the landsat files and stack only sentinel 2 bottom of atmosphere observations
raster_paths.extend(tifs)  # write the ones i need into my object to stack from
years = [int(re.search(r"\d{8}", raster_path).group(0)) for raster_path in raster_paths]  # i need to extract the date from each file path...
raster_paths = [raster_path for raster_path, year in zip(raster_paths, years) if year <= 20230302]  # ... to cut off at last image 20230302 as i trained my model with this sequence length, this might change in the future with transformers
datacube = [rxr.open_rasterio(raster_path) for raster_path in raster_paths]  # i user rxr because the clip function from rasterio sucks
# the datacube is too large for the RAM of my contingent so i need to subset using the 5x5km tiles
grid_tiles = glob.glob(os.path.join('/my_volume/FORCE_tiles_subset_BW/', '*' + tile + '*.gpkg'))

# checking workflow:

minitile = grid_tiles[0]
crop_shape = gpd.read_file(minitile)
clipped_datacube = []

for raster in datacube:  # clip the rasters using the geometry
    print(minitile)
    crop_shape = gpd.read_file(minitile)
    clipped_datacube = []  # empty object for adding clipped rasters
    try:
        print('yes')
        clipped_raster = raster.rio.clip(crop_shape.geometry.apply(mapping))
        clipped_datacube.append(clipped_raster)
    except:
        print('no')
        break

datacube_np = np.array(clipped_datacube) # this is now a clipped datacube for the first minitile
data_for_prediction = torch.tensor(datacube_np) # int16
data_for_prediction = data_for_prediction.to(torch.float32) # TODO: where did the sequence length go? it's 500x500x10x1

x = datacube_np.shape[2]  # i need the dimensions of the raster to create an empty one
y = datacube_np.shape[3]
# prediction = torch.zeros([x, y])  # create empty tensor where results will be written
# data_for_prediction = data_for_prediction.permute(2, 3, 0, 1)  # rearrange data

data = data_for_prediction.view(10, 500, -1) # this turns the data into a 3D tensor of rowsxcolumnsxbands
data_for_prediction = data.permute(2, 1, 0) # reorders

ct1 = datetime.datetime.now()
print("current time:-", ct1)
predictions = predict(data_for_prediction) # TODO: what does this represent? a row? a column?
ct = datetime.datetime.now()
print("current time:-", ct)
print("start time:-", ct1)
print('done')

predictions = torch.tensor(predictions)  # Convert the list to an array
map = predictions.numpy()


for i in range(x):
    input = data_for_prediction[i]
    outputs = model_pkl(input)
    _, predicted = torch.max(outputs.data, 1)
    prediction[i] = predicted
    print(i)