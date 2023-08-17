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
from rasterio.transform import from_origin # for assigning an origin to the created map

# tiles = np.loadtxt('/my_volume/BW_tiles.txt', dtype=str) # this file contains the XY tile names of my AOI in the same format as FORCE

device = torch.device('cpu') # assigning cpu for inference
model_pkl = torch.load('/my_volume/bi_lstm_demo.pkl',  map_location=torch.device('cpu')) # loading the trained model

def predict(input):
    outputs = model_pkl(input)
    _, predicted = torch.max(outputs.data, 1)
    return predicted

raster_paths = []

tile ="X0067_Y0058" # Landshut

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

tile ="X0067_Y0058" # Landshut
s2dir = glob.glob(('/force/FORCE/C1/L2/ard/' + tile + '/'), recursive=True)  # Initialize path to Sentinel-2 time series
raster_paths = []  # create object to be filled with rasters to be stacked
tifs = glob.glob(os.path.join(s2dir[0], "*SEN*BOA.tif"))  # i want to ignore the landsat files and stack only sentinel 2 bottom of atmosphere observations
raster_paths.extend(tifs)  # write the ones i need into my object to stack from
years = [int(re.search(r"\d{8}", raster_path).group(0)) for raster_path in raster_paths]  # i need to extract the date from each file path...
raster_paths = [raster_path for raster_path, year in zip(raster_paths, years) if year <= 20230302]  # ... to cut off at last image 20230302 as i trained my model with this sequence length, this might change in the future with transformers
datacube = [rxr.open_rasterio(raster_path) for raster_path in raster_paths]  # i user rxr because the clip function from rasterio sucks
# the datacube is too large for the RAM of my contingent so i need to subset using the 5x5km tiles
# grid_tiles = glob.glob(os.path.join('/my_volume/FORCE_tiles_subset_BW/', '*' + tile + '*.gpkg'))
grid_tiles = '/my_volume/31.gpkg'
# checking workflow:

minitile = grid_tiles
crop_shape = gpd.read_file(minitile)
clipped_datacube = [] # empty object for adding clipped rasters

iti = 1
total = str(len(datacube))
for raster in datacube:  # clip the rasters using the geometry
    crop_shape = gpd.read_file(minitile)
    print('cropping ' + str(iti) + ' out of ' + total)
    iti = iti + 1
    try:
        clipped_raster = raster.rio.clip(crop_shape.geometry.apply(mapping))
        clipped_datacube.append(clipped_raster)
    except:
        print('not working')
        break

datacube_np = np.array(clipped_datacube, ndmin = 4) # this is now a clipped datacube for the first minitile, fixing it to be 4 dimensions
datacube_np.shape
datacube_torch16 = torch.tensor(datacube_np) # turn the numpy array into a pytorch tensor, the result is in int16..
datacube_torch32 = datacube_torch16.to(torch.float32) # ...so we need to transfert it to float32 so that the model can use it as input
datacube_torch32.shape # torch.Size([320, 10, 500, 500])

data_for_prediction = datacube_torch32.permute(2, 3, 0, 1)  # rearrange data
data_for_prediction.shape #torch.Size([500, 500, 320, 10])

# data = data_for_prediction.reshape(320, 10, -1) # torch.Size([320, 10, 250000])

# data = data_for_prediction2.view(320, 10, -1)
# data = data_for_prediction2.view(10, 500, -1) # this turns the data into a 3D tensor of rowsxcolumnsxbands
# data_for_prediction_3d = data.permute(2, 1, 0) # reorders

# x = data_for_prediction_3d.shape[2]  # i need the dimensions of the raster to create an empty one
# y = data_for_prediction_3d.shape[3]
x = 500
y = 500

result = torch.zeros([x, y])  # create empty tensor where results will be written

for row in range(x):
    predictions = predict(data_for_prediction[row])
    print(row)
    result[row] = predictions

result = torch.tensor(result)  # Convert the list to an array
map = result.numpy()

crop_geometry = crop_shape.geometry.iloc[0]
crop_bounds = crop_geometry.bounds
origin = (crop_bounds[0], crop_bounds[3])  # Extract the origin from the bounds


# construct metadata from minitile and matrix
metadata = {
    'driver': 'GTiff',
    'width': map.shape[1],
    'height': map.shape[0],
    'count': 1,  # Number of bands
    'dtype': map.dtype,
    'crs': 'EPSG:3035',  # Set the CRS (coordinate reference system) code as needed
    'transform': from_origin(origin[0], origin[1], 10, 10)  # Set the origin and pixel size (assumes each pixel is 1 unit)
}

with rasterio.open(os.path.join('/my_volume/', 'landshut_example2.tif'), 'w', **metadata) as dst:
    dst.write_band(1, map.astype(rasterio.float32))