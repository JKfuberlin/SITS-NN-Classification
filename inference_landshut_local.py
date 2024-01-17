import torch.utils.data as Data # For dataloader class
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

def predict(data_for_prediction_loader):
    model_pkl.eval() # set model to eval mode to avoid dropout layer
    with torch.no_grad(): # do not track gradients during forward pass to speed up
        for (inputs) in data_for_prediction_loader:
            batch = inputs.to(device)  # put the data in gpu
            output_probabilities = model_pkl(batch)  # prediction
            _, predicted_class = torch.max(output_probabilities,1)  # retrieving the class with the highest probability after softmax
    return predicted_class
# the underscore means that the first entry of the torch.max function is discarded and only the second written into predicted

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device configuration

REMOTE = True

if REMOTE:
    model_pkl = torch.load('/point_storage/Transformer_1.pkl', map_location=torch.device('cpu'))  # loading the trained model
    raster_paths = []
    tile ="X0066_Y0056"
    tile ="X0067_Y0058" # Landshut
    s2dir = glob.glob(('/force/FORCE/C1/L2/ard/' + tile + '/'), recursive=True)  # Initialize path to Sentinel-2 time series
    raster_paths = []  # create object to be filled with rasters to be stacked
    tifs = glob.glob(os.path.join(s2dir[0], "*SEN*BOA.tif"))  # i want to ignore the landsat files and stack only sentinel 2 bottom of atmosphere observations

### get dates
    dates = [pd.to_datetime(s[35:43], format='%Y%m%d') for s in tifs]
    dates = pd.to_datetime(dates, format="%Y%m%d").sort_values() # sort ascending
    comparison_date = pd.to_datetime('20230302', format="%Y%m%d")

# Filter dates to exclude entries before 20230302
    dates = dates[dates <= comparison_date]

# these dates are the DOY that need to be passed to the forward method
### here, we need to retrieve and assign the correct DOY values
# get day of the year of startDate
    t0 = dates[0].timetuple().tm_yday
    input_dates = np.array([(date - dates[0]).days for date in dates]) + t0
    seq_len = len(input_dates)
    doy = np.zeros((seq_len,), dtype=int)
    DOY = input_dates

    raster_paths.extend(tifs)  # write the ones i need into my object to stack from
    years = [int(re.search(r"\d{8}", raster_path).group(0)) for raster_path in raster_paths]  # i need to extract the date from each file path...
    raster_paths = [raster_path for raster_path, year in zip(raster_paths, years) if year <= 20230302]  # ... to cut off at last image 20230302 as i trained my model with this sequence length, this might change in the future with transformers
    datacube = [rxr.open_rasterio(raster_path) for raster_path in raster_paths]  # i user rxr because the clip function from rasterio sucks
    # datacube.append(doy)# the datacube is too large for the RAM of my contingent so i need to subset using the 5x5km tiles
# grid_tiles = glob.glob(os.path.join('/my_volume/FORCE_tiles_subset_BW/', '*' + tile + '*.gpkg'))
    grid_tiles = '/point_storage/landshut_minibatch.gpkg'
# checking workflow:

    minitile = grid_tiles
    crop_shape = gpd.read_file(minitile)
    clipped_datacube = [] # empty object for adding clipped rasters

    i = 1
    total = str(len(datacube))
    for raster in datacube:  # clip the rasters using the geometry
        crop_shape = gpd.read_file(minitile)
        print('cropping ' + str(i) + ' out of ' + total)
        i = i + 1
        try:
            clipped_raster = raster.rio.clip(crop_shape.geometry.apply(mapping))
            clipped_datacube.append(clipped_raster)
        except:
            print('not working')
            break


else:
    print('running local')
    if torch.cuda.is_available() == False:
        model_pkl = torch.load('/point_storage/data/Transformer_1.pkl', map_location=torch.device('cpu'))  # loading the trained model
    # data_for_prediction = torch.load('/home/j/data/datacube_doy.pt')
    # clipped_datacube = np.load('/home/j/data/landshut_cropped_dc.npy')
    # DOY = pandas.read_csv('/home/j/data/doy_pixel_subset.csv', sep = '\t', header = None)
    DOY = pd.read_csv('/point_storage/data/doy_pixel_subset.csv', sep='\t', header=None)
    clipped_datacube = np.load('/point_storage/data/landshut_cropped_dc.npy')

DOY = np.array(DOY)
datacube_np = np.array(clipped_datacube, ndmin = 4) # this is now a clipped datacube for the first minitile, fixing it to be 4 dimensions
# Reshape doy to have a new axis
doy_reshaped = DOY.reshape((329, 1, 1, 1))

# Repeat the doy values along the third and fourth dimensions to match datacube_np
doy_reshaped = np.repeat(doy_reshaped, 500, axis=2)
doy_reshaped = np.repeat(doy_reshaped, 500, axis=3)

# Concatenate along the second axis
datacube_np = np.concatenate((datacube_np, doy_reshaped), axis=1)

# datacube_np.shape
datacube_torch16 = torch.tensor(datacube_np) # turn the numpy array into a pytorch tensor, the result is in int16..
datacube_torch32 = datacube_torch16.to(torch.float32) # ...so we need to transfer it to float32 so that the model can use it as input
# datacube_torch32.shape # torch.Size([320, 10, 500, 500])

data_for_prediction = datacube_torch32.permute(2, 3, 0, 1)  # rearrange data
data_for_prediction.shape #torch.Size([500, 500, 320, 10])
# [500, 500, 329, 10] after DOY concatentation, expected [500, 500, 329, 11]

'''now i need to turn the datacube data into the DataLoader format to enable prediction'''
data_for_prediction_loader = Data.DataLoader(data_for_prediction, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

x = 500
y = 500
result = np.zeros((x, y, 1))
for row in range(x):
    for col in range(x):
        pixel = data_for_prediction[row, col]
        data_for_prediction_loader = Data.DataLoader(pixel, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
        predicted_class = predict(data_for_prediction_loader)
        result[row, col, :] = predicted_class

# result = torch.tensor(result)  # Convert the list to an array
map = result

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

with rasterio.open(os.path.join('/point_storage/', 'landshut_transformer.tif'), 'w', **metadata) as dst:
    # dst.write_band(1, map.astype(rasterio.float32))
    dst.write(map.astype(rasterio.float32), indexes=1)


# map_reshaped = map.transpose(1, 2, 0, 3)
#
# with rasterio.open(os.path.join('/point_storage/', 'landshut_transformer.tif'), 'w', **metadata) as dst:
#     for i in range(map_reshaped.shape[3]):
#         dst.write_band(i + 1, map_reshaped[:, :, :, i].astype(rasterio.float32))
