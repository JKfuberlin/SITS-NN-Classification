# import tifffile as tiff
# import numpy as np
#
# # Load the TIFF file
# image_path = '/home/j/data/s2test.tif'
# image = tiff.imread(image_path)
#
# # Convert the image to a NumPy array
# image_array = np.array(image)
#
# print(image_array)
# # Now you can use the image_array for prediction with your trained model
#

import os
import glob
import re
import rasterio
import geopandas as gpd
import numpy as np

# Set the EPSG code
epsg_code = 3035

# Read shapefiles
shapes = gpd.read_file('/home/eouser/shapes/FEpoints10m_3035.gpkg')

# Get unique Tile IDs
tiles = shapes['Tile_ID'].unique()

# Initialize path to Sentinel-2 time series
s2dirs = glob.glob('/force/FORCE/C1/L2/ard/**', recursive=True)

# Process each tile (only the first tile in this code snippet)
tile = tiles[0]
g = str(tile)
s2dirs_aoi = [s2dir for s2dir in s2dirs if re.search(g, s2dir)]

raster_paths = []
# Pattern for regex: 20221127_LEVEL2_SEN2A_BOA.tif
# Get tifs from each folder
for s2dir in s2dirs_aoi:
    tifs = glob.glob(os.path.join(s2dir, "*SEN*BOA.tif"))
    raster_paths.extend(tifs)

raster_paths2 = []

# Cut off at last image 20230302
# Extract year from filenames
years = [int(re.search(r"\d{8}", raster_path).group(0)) for raster_path in raster_paths]

# Filter filenames based on the condition
raster_paths = [raster_path for raster_path, year in zip(raster_paths, years) if year <= 20230302]

# Create a vector for renaming the datacube
for path in raster_paths:
    for band in range(1, 11):
        c = f"{path}{band}"
        raster_paths2.append(c)

# Load raster data as a datacube
datacube = np.stack([rasterio.open(raster_path).read() for raster_path in raster_paths], axis=-1)

print(f"Loading rasters {g}")
# Rename raster names to include time
datacube_names = dict(zip(range(1, 11), raster_paths2))
datacube = datacube.rename(datacube_names, axis=-1)

# Extract bounding box from datacube and convert it to a GeoDataFrame
bbox = shapes.total_bounds
b = gpd.GeoSeries(gpd.GeoDataFrame(geometry=[gpd.box(*bbox)]))
b.crs = f"EPSG:{epsg_code}"
shapes_extract = gpd.sjoin(shapes, b, how='inner', op='intersects')
print(len(shapes_extract))

# Convert datacube raster to a matrix
mymat = datacube.reshape((-1, datacube.shape[-1]))