import os # for
import glob # for grepping i guess
import re # for
import rasterio # for reading rasters
import geopandas as gpd # for
import numpy as np # for
import datetime # for benchmarking
import torch # for loading the model and actual inference
import rioxarray as rxr # for raster clipping

import multiprocessing # for parallelization
from models.lstm import LSTMClassifier

import sys # for getting object size
from shapely.geometry import mapping


tile = 'X0058_Y0056'
epsg_code = 3035 # Set the EPSG code
shapes = gpd.read_file('/home/eouser/shapes/FEpoints10m_3035.gpkg') # Read shapefiles
tiles = shapes['Tile_ID'].unique() # Get unique Tile IDs for iteration
s2dirs_aoi = glob.glob(('/force/FORCE/C1/L2/ard/'+tile), recursive=True) # Initialize path to Sentinel-2 time series TODO: make this iterable by reading the last part from tile IDs

raster_paths = [] # create object to be filled with rasters to be stacked
for s2dir in s2dirs_aoi: # Get tifs from each folder
    tifs = glob.glob(os.path.join(s2dir, "*SEN*BOA.tif")) # i want to ignore the landsat files and stack only sentinel 2 bottom of atmosphere observations
    raster_paths.extend(tifs) # write the ones i need into my object to stack from

years = [int(re.search(r"\d{8}", raster_path).group(0)) for raster_path in raster_paths] # i need to extract the date from each file path...
raster_paths = [raster_path for raster_path, year in zip(raster_paths, years) if year <= 20230302] # ... to cut off at last image 20230302 as i trained my model with this sequence length, this might change in the future with transformers


# the datacube is too large for the RAM of my contingent so i need to subset, i first try the workflow with an AOI:


# np.save('/my_volume/clipped_datacube.npy', clipped_datacube)
# clipped_datacube = np.load('/my_volume/clipped_datacube.npy')
device = torch.device('cpu')
model_pkl = torch.load('/my_volume/bi_lstm_demo.pkl',  map_location=torch.device('cpu'))

datacube = [rxr.open_rasterio(raster_path) for raster_path in raster_paths]
# datacube = [rasterio.open(raster_path) for raster_path in raster_paths]

#should be [number of samples, sequence length, number of bands]
# i have sequence length, num bands, x, y need to flatten the data
# squeeze/unsqueeze / transpose
datacube_np = np.stack(datacube)

np.save('/my_volume/datacube_np.npy', datacube_np)
datacube_np = np.load('/my_volume/datacube_np.npy')

# datacube_np = np.stack(clipped_datacube, axis=-1) # turn the datacube into a numpy array so it can be turnt into a tensor for inference

# torch.Size([10, 320, 202, 203])

# data_for_prediction = torch.tensor(datacube_np) # int16
# data_for_prediction = data_for_prediction.to(torch.float32) # this crashes the python env

data_for_prediction = torch.cat(processed_chunks, dim=0)

# data = data_for_prediction.view(10, 320, -1)# flattening last two
# data_for_prediction = data.permute(2, 1, 0)
# the output size should be (number of pixels = 203 * 202, sequence length = 320, number of bands = 10)

# i need to add one dimension where the prediction is written into
# flatten the image and turn all rows of pixels into a sequence so i don't have the 203*202 problem.

# try dataset = Data.TensorDataset(data_for_prediction)
x=clipped_datacube.shape[2]
y=clipped_datacube.shape[3]
# minibatch_size = 512
# batch_num = (x*y)/minibatch_size
# iterator = 0
#
# for batch in [0:batch_num]

# create empty tensor to fill with outputs
prediction = torch.zeros([x,y])
# rearrange data
data_for_prediction = data_for_prediction.permute(2, 3, 0, 1)

def predict(input):
    outputs = model_pkl(input)
    _, predicted = torch.max(outputs.data, 1)
    return predicted

# Define the number of cores to use
num_cores = 15

# Create a multiprocessing pool
pool = multiprocessing.Pool(processes=num_cores)

# Iterate over the range in parallel and store the results in a list
predictions = pool.map(predict, data_for_prediction)

# Close the pool to release resources
pool.close()
pool.join()

# Convert the list to an array if desired
predictions = torch.tensor(predictions)

map = prediction.numpy()
ct1 = datetime.datetime.now()
print("current time:-", ct1)
# now the numpy array needs a crs and so on to be saved as a GEOTiff
# first i retrieve the metadata from the original raster
with rasterio.open(raster_paths[0]) as src: # i can use any image from the stack cuz they all the same
    metadata = src.meta
    # Update the number of bands in the metadata to match the modified image
    metadata['count'] = 1

with rasterio.open('/my_volume/map_BILSTM_X0058_Y0056.tif', 'w', **metadata) as dst: # open write connection to file using the metadata
    dst.write(map, 1)
ct = datetime.datetime.now()
print("current time:-", ct)
print("start time:-", ct1)
print('done')


# minibatch = data_for_prediction[0:512]
# datacube should now be a numpy array and ready for inference


# i need to send model and dataset.to(device)
# and i need to modify the model architecture to accept cpu instead of gpu
#


# # bugfixing
# num_bands = 12
# input_size = 16
# hidden_size = 32
# num_layers = 1
# num_classes = 9
# bidirectional = True

num_bands = 10
input_size = 64
hidden_size = 128
num_layers = 3
num_classes = 10
bidirectional = True

# TODO review which pth file is written on gromit
from lstm  import LSTMCPU
model_pth = LSTMCPU(num_bands, input_size, hidden_size, num_layers, num_classes, bidirectional).to(device)
model_pth.load_state_dict(torch.load('/my_volume/bi_lstm_demo.pth',  map_location=torch.device('cpu')))
prediction = model_pth(minibatch)

from lstm import LSTMClassifier
from models.lstm import LSTMClassifier


# model = LSTMClassifier(num_bands, input_size, hidden_size, num_layers, num_classes, bidirectional).to(device)
# model.load_state_dict(torch.load('/my_volume/bi_lstm_demo.pth',  map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('/my_volume/bi_lstm_demo.pkl',  map_location=torch.device('cpu')))
#
# with open('/my_volume/bi_lstm_demo.pkl', 'rb') as f:
#     loaded_classifier = pickle.load(f)



# model = torch.load('/my_volume/bi_lstm_demo.pkl',  map_location=torch.device('cpu'))
# pickled_model = pickle.load(open('/my_volume/bi_lstm_demo.pth', 'rb'))

datacube = np.load('/my_volume/datacube.npy')




#x_set = torch.from_numpy(datacube)

# Annex: here i put all the code that might be useful once again

# ANNEX I - # i am not sure if i need to rename the datacube layers to fit what the model is trained on -> Annex I
# # Create a vector for renaming the datacube
# for path in raster_paths:
#     for band in range(1, 11):
#         c = f"{path}{band}"
#         raster_paths2.append(c)
# END I

# ANNEX II reading in rasters
# approach tifffile
# import tifffile as tiff
# # using datetime module
# import datetime
#
# # ct stores current time
# ct = datetime.datetime.now()
# print("current time:-", ct)
# images = [tiff.imread(raster_path) for raster_path in raster_paths]
# ct = datetime.datetime.now()
# print("current time:-", ct)
# stacked_image = np.stack(images, axis=-1)
# cropped_image = stacked_image[:, ymin:ymax, xmin:xmax]



# approach opencv
# images = [cv2.imread(raster_path, cv2.IMREAD_UNCHANGED) for raster_path in raster_paths]
# stacked_image = np.stack(images, axis=-1)
# cropped_data1 = stacked_image[:, ymin:ymax, xmin:xmax]
# cropped_data2 = stacked_image[ymin:ymax, xmin:xmax, :]
# cropped_data3 = stacked_image[ymin:ymax, xmin:xmax,]
#
# # approach rio.stack
# datacube_rio = stack(raster_paths)
#
# # approach earthpy
# import earthpy as et
# import earthpy.spatial as es
# datacube_earthpy = es.stack(raster_paths)
# def crop_raster(datacube_np, xmin, ymin, xmax, ymax):
#     cropped_data = datacube_np[:, ymin:ymax, xmin:xmax]
#     return cropped_data
#
# # custom earthpy approach
# from custom import custom_stack
# test = custom_stack(raster_paths)
# END II

# ANNEX III
# read a shapefile with fiona:
# with fiona.open('/my_volume/shapes/aoi_Hardtwald/aoi_Hardtwald.shp', "r") as shapefile: # load the AOI for this test, there also is a gpkg
#     crop_shape = [feature["geometry"] for feature in shapefile]
# END III

# ANNEX IV - clipping the datacube to AOI extent:
#
# shapefile_path = '/my_volume/shapes/aoi_Hardtwald/aoi_Hardtwald_3035.gpkg'
# crop_shape = gpd.read_file(shapefile_path)
#
# # i user rxr because the clip function from rasterio sucks
# datacube2 = [rxr.open_rasterio(raster_path) for raster_path in raster_paths]
# compare = rasterio.open(raster_paths[1])
#
# clipped_datacube = []
# import datetime
# ct1 = datetime.datetime.now()
# print("current time:-", ct1)
# for raster in datacube2:
#     clipped_raster = raster.rio.clip(crop_shape.geometry.apply(mapping))
#     print('.')
#     clipped_datacube.append(clipped_raster)
#
#
# ct = datetime.datetime.now()
# print("current time:-", ct)
# print("start time:-", ct1)
# END IV


# ANNEX V
# other approaches to stacking the raster
# # approach rio.stack
# datacube = stack(raster_paths)
#
#
# # approach I
# # Read metadata of first file
# with rasterio.open(raster_paths[0]) as src0:
#     meta = src0.meta
#
# # Update meta to reflect the number of layers
# meta.update(count = len(raster_paths))
#
# # Read each layer and write it to stack
# with rasterio.open('stack.tif', 'w', **meta) as dst:
#     for id, layer in enumerate(raster_paths, start=1):
#         with rasterio.open(layer) as src1:
#             dst.write_band(id, src1.read(1))



# ANNEX VI - trying to chunk down tensor transformation
#
#
# chunk_size = 1  # Adjust this value based on available memory and array size
# # num_samples = datacube_np.shape[0] # is this right?
# num_samples = 3000 # the datacube apparently is 3000 x 3000 in dimension
# processed_chunks = []
#
# # chunk = datacube_np[0:10]
# # chunk_tensor = torch.tensor(chunk)
# # chunk = chunk_tensor.to(torch.float32)
# # processed_chunks.append(chunk_tensor)
#
# ct1 = datetime.datetime.now()
# print("current time:-", ct1)
#
# for i in range(0, num_samples, chunk_size):
#     chunk = datacube_np[i:i+chunk_size]
#     chunk_tensor = torch.tensor(chunk)
#     chunk = chunk_tensor.to(torch.float32)
#     processed_chunks.append(chunk_tensor)
#     # Perform incremental concatenation if there are more than 1 chunks
#     while len(processed_chunks) > 1:
#         new_chunks = []
#         for j in range(0, len(processed_chunks), 2):
#             if j+1 < len(processed_chunks):
#                 combined_chunk = torch.cat([processed_chunks[j], processed_chunks[j+1]], dim=0)
#                 new_chunks.append(combined_chunk)
#             else:
#                 new_chunks.append(processed_chunks[j])
#         processed_chunks = new_chunks
#
# # The single remaining chunk is the final concatenated tensor
# data_for_prediction = processed_chunks[0]
#
# ct = datetime.datetime.now()
# print("current time:-", ct)
# print("start time:-", ct1)

# END VI


#
# datacube = [rasterio.open(raster_path).read() for raster_path in raster_paths] # Load raster data as a datacube, unfortrunately this creates a list that i cannot crop
# for raster in datacube:
#     raster = raster.rasterio.clip(crop_shape.geometry.apply(mapping),
#                                       # This is needed if your GDF is in a diff CRS than the raster data
#                                       crop_shape.crs)
#
# # data_crop = rasterio.mask.mask(datacube, crop_shape, crop = True)
#
#
# data_crop = datacube.rasterio.clip(crop_shape.geometry.apply(mapping),
#                                       # This is needed if your GDF is in a diff CRS than the raster data
#                                       crop_shape.crs)
#
#
# datacube_np = np.stack(data_crop, axis=-1) # turn the datacube into a numpy array so it can be turnt into a tensor for inference
#

# ANNEX who cares:
# trying to crop using coordinates

# xmin, ymin, xmax, ymax = crop_shape.total_bounds # getting the bounds of the shapefile of my aoi
#
# # yes, i really don't know how to do this in a smart way, judge me
# xmin = int(xmin)
# xmax = int(xmax)
# ymin = int(ymin)
# ymax = int(ymax)




# #saving
# kwargs = datacube2[2].meta
# kwargs.update(
#     dtype=rasterio.float32,
#     count=1,
#     compress='lzw')
#
# with rasterio.open(os.path.join('/my_volume/', 'example.tif'), 'w', **kwargs) as dst:
#     dst.write_band(1, datacube2.astype(rasterio.float32))
#
#
#
# with rasterio.Env():
#     profile = datacube2[2].profile
#     profile.update(
#         dtype=rasterio.uint8,
#         count=1,
#         compress='lzw')
#     with rasterio.open('/my_volume/example.tif', 'w', **profile) as dst:
#         dst.write(datacube2.astype(rasterio.uint8), 1)

# a = rasterio.open(raster_paths[0])
# >>> a.bounds
# BoundingBox(left=4196026.3630416505, bottom=2864919.6079648044, right=4226026.3630416505, top=2894919.6079648044)
# while
# >>> crop_shape.geometry[0]
# <POLYGON ((457870.222 5437357.401, 459870.222 5437357.401, 459870.222 543535...>

# sys.getsizeof(datacube_np)
# # datacube_cropped = crop_raster(datacube_np, xmin, ymin, xmax, ymax)
# cropped_data = datacube_np[:, ymin:ymax, xmin:xmax]
# cropped_data = datacube_np[ymin:ymax, xmin:xmax]
# sys.getsizeof(cropped_data)

# need to transform datacube numpy array into tensor
# data_for_prediction = datacube.reshape(-1, num_bands)