# TODO fix GPU inference: https://stackoverflow.com/questions/71278607/pytorch-expected-all-tensors-on-same-device
# TODO script should take path to FORCE tile as input parameter, not query dirs on its own
import argparse
from pathlib import Path
from re import search
from typing import List, Any, Union, Dict
import numpy as np
import rasterio
import rioxarray as rxr
import torch
import xarray
import logging
from time import time
from sits_classifier import models

parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Run inference with already trained LSTM classifier on a remote-sensing time series represented as "
                "FORCE ARD datacube.")
parser.add_argument("-w", "--weights", dest="weights", required=True, type=Path,
                    help="Path to pre-trained classifier to be loaded via `torch.load`. Can be either a relative or "
                         "absolute file path.")
parser.add_argument("--input-tiles", dest="input", required=True, type=Path,
                    help="List of FORCE tiles which should be used for inference. Each line should contain one FORCE "
                         "tile specifier (Xdddd_Ydddd).")
parser.add_argument("--input-dir", dest="base", required=True, type=Path,
                    help="Path to FORCE datacube.")
parser.add_argument("--input-glob", dest="iglob", required=False, type=str, default="*",
                    help="Optional glob pattern to restricted files used from `input-dir`.")
parser.add_argument("--output-dir", dest="out", required=True, type=Path,
                    help="Path to directory into which predictions should be saved.")
parser.add_argument("--date-cutoff", dest="date", required=True, type=int,
                    help="Cutoff date for time series which should be included in datacube for inference.")
parser.add_argument("--mask-dir", dest="masks", required=False, type=Path, default=None,
                    help="Path to directory containing folders in FORCE tile structure storing "
                         "binary masks with a value of 1 representing pixels to predict. Others can be nodata "
                         "or 0. Masking is done on a row-by-row basis. I.e., the entire unmasked datacube "
                         "is constructed from the files found in `input-dir`. Only when handing a row of "
                         "pixels to the DL-model for inference are data removed. Thus, this does not reduce "
                         "the memory footprint, but can speed up inference significantly under certain "
                         "conditions.")
parser.add_argument("--mask-glob", dest="mglob", required=False, type=str, default="mask.tif",
                    help="Optional glob pattern to restricted file used from `mask-dir`.")
parser.add_argument("--row-size", dest="row-block", required=False, type=int, default=None,
                    help="Row-wise size to read in at once. If not specified, query dataset for block size and assume "
                         "constant block sizes across all raster bands in case of multilayer files. Contrary to "
                         "what GDAL allows, if the entire raster extent is not evenly divisible by the block size, "
                         "an error will be raised and the process aborted. If only `row-size` is given, read the "
                         "specified amount of rows and however many columns are given by the datasets block size. "
                         "If both `row-size` and `col-size` are given, read tiles of specified size.")
parser.add_argument("--col-size", dest="col-block", required=False, type=int, default=None,
                    help="Column-wise size to read in at once. If not specified, query dataset for block size and "
                         "assume constant block sizes across all raster bands in case of multilayer files. Contrary to "
                         "what GDAL allows, if the entire raster extent is not evenly divisible by the block size, "
                         "an error will be raised and the process aborted. If only `col-size` is given, read the "
                         "specified amount of columns and however many rows are given by the datasets block size. "
                         "If both `col-size` and `row-size` are given, read tiles of specified size.")
parser.add_argument("--log", dest="log", required=False, action="store_true",
                    help="Emit logs?")
parser.add_argument("--log-file", dest="log-file", required=False, type=str,
                    help="If logging is enabled, write to this file. If omitted, logs are written to stdout.")

cli_args: Dict[str, Union[Path, int, bool, str]] = vars(parser.parse_args())

if cli_args.get("log"):
    if cli_args.get("log-file"):
        logging.basicConfig(level=logging.INFO, filename=cli_args.get("log-file"))
    else:
        logging.basicConfig(level=logging.INFO)

with open(cli_args.get("input"), "rt") as f:
    FORCE_tiles: List[str] = [tile.replace("\n", "") for tile in f.readlines()]

lstm: torch.nn.LSTM = torch.load(cli_args.get("weights"), map_location=torch.device('cpu')).eval()


def predict(model, data: torch.tensor) -> Any:
    """
    Apply previously trained LSTM to new data
    :param model: previously trained model
    :param torch.tensor data: new input data
    :return Any: Array of predictions
    """
    # TODO https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html suggests that
    #      multi-threading is possible and I observe multi-CPU usage.
    #      Is multi-threading on by default? Documentation suggests no, but I'm not sure.
    #      torch.__config__.parallel_info() suggests, that multi-threading is active!
    with torch.no_grad():
        outputs = model(data)
    _, predicted = torch.max(outputs.data, 1)
    return predicted


for tile in FORCE_tiles:
    start: float = time()
    logging.info(f"Processing FORCE tile {tile}")
    s2_tile_dir: Path = cli_args.get("base") / tile
    tile_paths: List[str] = [str(p) for p in s2_tile_dir.glob(cli_args.get("iglob"))]
    cube_inputs: List[str] = [
        tile_path for tile_path in tile_paths if int(search(r"\d{8}", tile_path).group(0)) <= cli_args.get("date")
    ]

    with rasterio.open(cube_inputs[0]) as f:
        metadata = f.meta
        metadata["count"] = 1
        metadata["dtype"] = rasterio.uint8
        metadata["nodata"] = 0
        row_block, col_block = f.block_shapes[0]

    tile_rows: int = metadata["height"]
    tile_cols: int = metadata["width"]
    output_torch: torch.tensor = torch.zeros([tile_rows, tile_cols], dtype=torch.long)

    row_step: int = cli_args.get("row-block") or row_block
    col_step: int = cli_args.get("col-block") or col_block

    if tile_rows % row_step != 0 or tile_cols % col_step != 0:
        raise AssertionError("Rows and columns must be divisible by their respective step sizes without remainder.")

    logging.info(f"Processing tile {tile = } in chunks of {row_step = } and {col_step = }")

    for row in range(0, tile_rows, row_step):
        for col in range(0, tile_cols, col_step):
            start_chunked: float = time()
            logging.info(f"Creating chunked data cube")
            s2_cube: Union[xarray.Dataset, xarray.DataArray, list[xarray.Dataset]] = []
            for cube_input in cube_inputs:
                ds: Union[xarray.Dataset, xarray.DataArray] = rxr.open_rasterio(cube_input)
                clipped_ds = ds.isel(y=slice(row, row + row_step),
                                     x=slice(col, col + col_step))
                s2_cube.append(clipped_ds)
                ds.close()

            logging.info(f"Converting chunked data cube to numpy array")
            s2_cube_np: np.ndarray = np.array(s2_cube, ndmin=4, dtype=np.float32)
            logging.info("Transposing Numpy array")
            s2_cube_npt: np.ndarray = np.transpose(s2_cube_np, (2, 3, 0, 1))

            del s2_cube
            del s2_cube_np

            """
            The code below uses masked tensors to mask the actual data cube instead of dropping unwanted observations
            during prediction. However, at the time of writing, the method ADDMM is not implemented for masked tensors
            in torch. Thus, inference fails when the model used, such as a lstm, calls this method somewhere.
            It could be worth revisiting this approach in the future as it may be more generalizable and potentially
            reduce memory usage.
            
            Additionally, masked tensors don't play well view tensor views. Thus, during inference, actual data copying
            is necessary (s2_cube_torch[chunk_rows].contiguous()). But since this is only done on a row by row basis, the actual impact memory wise should
            be negligible.
            
            # TODO dont hardcode filename and zero index
            mask_path: str = [str(p) for p in (cli_args.get("masks") / tile).glob("mask.tif")][0]
            with rxr.open_rasterio(mask_path) as ds:
                mask_ds: xarray.Dataset = ds.isel(y=slice(row, row + row_step),
                                                  x=slice(col, col + col_step))
                a: np.ndarray = np.array(mask_ds, ndmin=2, dtype=np.bool_)
                b: np.ndarray = np.expand_dims(a, axis=0)
                c: np.ndarray = np.repeat(b, 10, axis=1)
                d: np.ndarray = np.repeat(c, 323, axis=0)
                del mask_ds, a, b, c
            masked_array: torch.Tensor = torch.reshape(torch.from_numpy(d), (100, 100, 323, 10))  # no views, errors otherwise
            logging.info("Creating masked tensor")
            s2_cube_torch: Union[torch.Tensor, torch.masked.masked_tensor] = torch.masked.as_masked_tensor(
                torch.from_numpy(s2_cube_npt),
                masked_array
            )
            """
            if cli_args.get("masks"):
                mask_path: str = [str(p) for p in (cli_args.get("masks") / tile).glob(cli_args.get("mglob"))][0]
                with rxr.open_rasterio(mask_path) as ds:
                    mask_ds: xarray.Dataset = ds.isel(y=slice(row, row + row_step),
                                                      x=slice(col, col + col_step))
                    mask: np.ndarray = np.squeeze(np.array(mask_ds, ndmin=2, dtype=np.bool_), axis=0)
                    del mask_ds

            logging.info(f"Converting chunked numpy array to torch tensor")
            s2_cube_torch: Union[torch.Tensor, torch.masked.masked_tensor] = torch.from_numpy(s2_cube_npt)

            if cli_args.get("mask-dir"):
                merged_row: torch.Tensor = torch.zeros(col_step, dtype=torch.long)
                for chunk_rows in range(0, row_step):
                    merged_row.zero_()
                    squeezed_row: torch.Tensor = predict(
                        lstm,
                        # lines below are achieve the same subset
                        # torch.index_select(
                        #     s2_cube_torch[chunk_rows],
                        #     0,
                        #     torch.from_numpy(
                        #         np.indices((1, mask.shape[1]), dtype=np.int32)[1, 0, mask[chunk_rows]])))
                        s2_cube_torch[chunk_rows, mask[chunk_rows]])
                    merged_row[mask[chunk_rows]] = squeezed_row
                    output_torch[row + chunk_rows, col:col + col_step] = merged_row
            else:
                for chunk_rows in range(0, row_step):
                    output_torch[row + chunk_rows, col:col + col_step] = predict(lstm, s2_cube_torch[chunk_rows])

            logging.info(f"Processed chunk in {time() - start_chunked:.2f} seconds")

            del s2_cube_torch

    output_numpy: np.array = output_torch.numpy()

    with rasterio.open(cli_args.get("out") / (tile + ".tif"), 'w', **metadata) as dst:
        dst.write_band(1, output_numpy.astype(rasterio.uint8))

    logging.info(f"Processed tile {tile} in {time() - start:.2f} seconds")

    del metadata
    del output_torch
    del output_numpy

'''local'''
# import torch.utils.data as Data # For dataloader class
# import os # for general operating system functionality
# import glob # for retrieving files using strings/regex
# import re # for regex
# import rasterio # for reading rasters
# import geopandas as gpd # for reading shapefiles
# import numpy as np
# import datetime # for benchmarking
# import torch # for loading the model and actual inference
# import rioxarray as rxr # for raster clipping
# import multiprocessing # for parallelization
# from shapely.geometry import mapping # for clipping
# from rasterio.transform import from_origin # for assigning an origin to the created map
#
# def predict(data_for_prediction_loader):
#     model_pkl.eval() # set model to eval mode to avoid dropout layer
#     with torch.no_grad(): # do not track gradients during forward pass to speed up
#         for (inputs) in data_for_prediction_loader:
#             batch = inputs.to(device)  # put the data in gpu
#             output_probabilities = model_pkl(batch)  # prediction
#             _, predicted_class = torch.max(output_probabilities,1)  # retrieving the class with the highest probability after softmax
#     return predicted_class
# # the underscore means that the first entry of the torch.max function is discarded and only the second written into predicted
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device configuration
#
# REMOTE = True
#
# if REMOTE:
#     model_pkl = torch.load('/point_storage/Transformer_1.pkl', map_location=torch.device('cpu'))  # loading the trained model
#     raster_paths = []
#     tile ="X0066_Y0056"
#     tile ="X0067_Y0058" # Landshut
#     s2dir = glob.glob(('/force/FORCE/C1/L2/ard/' + tile + '/'), recursive=True)  # Initialize path to Sentinel-2 time series
#     raster_paths = []  # create object to be filled with rasters to be stacked
#     tifs = glob.glob(os.path.join(s2dir[0], "*SEN*BOA.tif"))  # i want to ignore the landsat files and stack only sentinel 2 bottom of atmosphere observations
#
# ### get dates
#     dates = [pd.to_datetime(s[35:43], format='%Y%m%d') for s in tifs]
#     dates = pd.to_datetime(dates, format="%Y%m%d").sort_values() # sort ascending
#     comparison_date = pd.to_datetime('20230302', format="%Y%m%d")
#
# # Filter dates to exclude entries before 20230302
#     dates = dates[dates <= comparison_date]
#
# # these dates are the DOY that need to be passed to the forward method
# ### here, we need to retrieve and assign the correct DOY values
# # get day of the year of startDate
#     t0 = dates[0].timetuple().tm_yday
#     input_dates = np.array([(date - dates[0]).days for date in dates]) + t0
#     seq_len = len(input_dates)
#     doy = np.zeros((seq_len,), dtype=int)
#     DOY = input_dates
#
#     raster_paths.extend(tifs)  # write the ones i need into my object to stack from
#     years = [int(re.search(r"\d{8}", raster_path).group(0)) for raster_path in raster_paths]  # i need to extract the date from each file path...
#     raster_paths = [raster_path for raster_path, year in zip(raster_paths, years) if year <= 20230302]  # ... to cut off at last image 20230302 as i trained my model with this sequence length, this might change in the future with transformers
#     datacube = [rxr.open_rasterio(raster_path) for raster_path in raster_paths]  # i user rxr because the clip function from rasterio sucks
#     # datacube.append(doy)# the datacube is too large for the RAM of my contingent so i need to subset using the 5x5km tiles
# # grid_tiles = glob.glob(os.path.join('/my_volume/FORCE_tiles_subset_BW/', '*' + tile + '*.gpkg'))
#     grid_tiles = '/point_storage/landshut_minibatch.gpkg'
# # checking workflow:
#
#     minitile = grid_tiles
#     crop_shape = gpd.read_file(minitile)
#     clipped_datacube = [] # empty object for adding clipped rasters
#
#     i = 1
#     total = str(len(datacube))
#     for raster in datacube:  # clip the rasters using the geometry
#         crop_shape = gpd.read_file(minitile)
#         print('cropping ' + str(i) + ' out of ' + total)
#         i = i + 1
#         try:
#             clipped_raster = raster.rio.clip(crop_shape.geometry.apply(mapping))
#             clipped_datacube.append(clipped_raster)
#         except:
#             print('not working')
#             break
#
#
# else:
#     print('running local')
#     if torch.cuda.is_available() == False:
#         model_pkl = torch.load('/point_storage/data/Transformer_1.pkl', map_location=torch.device('cpu'))  # loading the trained model
#     # data_for_prediction = torch.load('/home/j/data/datacube_doy.pt')
#     # clipped_datacube = np.load('/home/j/data/landshut_cropped_dc.npy')
#     # DOY = pandas.read_csv('/home/j/data/doy_pixel_subset.csv', sep = '\t', header = None)
#     DOY = pd.read_csv('/point_storage/data/doy_pixel_subset.csv', sep='\t', header=None)
#     clipped_datacube = np.load('/point_storage/data/landshut_cropped_dc.npy')
#
# DOY = np.array(DOY)
# datacube_np = np.array(clipped_datacube, ndmin = 4) # this is now a clipped datacube for the first minitile, fixing it to be 4 dimensions
# # Reshape doy to have a new axis
# doy_reshaped = DOY.reshape((329, 1, 1, 1))
#
# # Repeat the doy values along the third and fourth dimensions to match datacube_np
# doy_reshaped = np.repeat(doy_reshaped, 500, axis=2)
# doy_reshaped = np.repeat(doy_reshaped, 500, axis=3)
#
# # Concatenate along the second axis
# datacube_np = np.concatenate((datacube_np, doy_reshaped), axis=1)
#
# # datacube_np.shape
# datacube_torch16 = torch.tensor(datacube_np) # turn the numpy array into a pytorch tensor, the result is in int16..
# datacube_torch32 = datacube_torch16.to(torch.float32) # ...so we need to transfer it to float32 so that the model can use it as input
# # datacube_torch32.shape # torch.Size([320, 10, 500, 500])
#
# data_for_prediction = datacube_torch32.permute(2, 3, 0, 1)  # rearrange data
# data_for_prediction.shape #torch.Size([500, 500, 320, 10])
# # [500, 500, 329, 10] after DOY concatentation, expected [500, 500, 329, 11]
#
# '''now i need to turn the datacube data into the DataLoader format to enable prediction'''
# data_for_prediction_loader = Data.DataLoader(data_for_prediction, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
#
# x = 500
# y = 500
# result = np.zeros((x, y, 1))
# for row in range(x):
#     for col in range(x):
#         pixel = data_for_prediction[row, col]
#         data_for_prediction_loader = Data.DataLoader(pixel, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
#         predicted_class = predict(data_for_prediction_loader)
#         result[row, col, :] = predicted_class
#
# # result = torch.tensor(result)  # Convert the list to an array
# map = result
#
# crop_geometry = crop_shape.geometry.iloc[0]
# crop_bounds = crop_geometry.bounds
# origin = (crop_bounds[0], crop_bounds[3])  # Extract the origin from the bounds
#
# # construct metadata from minitile and matrix
# metadata = {
#     'driver': 'GTiff',
#     'width': map.shape[1],
#     'height': map.shape[0],
#     'count': 1,  # Number of bands
#     'dtype': map.dtype,
#     'crs': 'EPSG:3035',  # Set the CRS (coordinate reference system) code as needed
#     'transform': from_origin(origin[0], origin[1], 10, 10)  # Set the origin and pixel size (assumes each pixel is 1 unit)
# }
#
# with rasterio.open(os.path.join('/point_storage/', 'landshut_transformer.tif'), 'w', **metadata) as dst:
#     # dst.write_band(1, map.astype(rasterio.float32))
#     dst.write(map.astype(rasterio.float32), indexes=1)
#
#
# # map_reshaped = map.transpose(1, 2, 0, 3)
# #
# # with rasterio.open(os.path.join('/point_storage/', 'landshut_transformer.tif'), 'w', **metadata) as dst:
# #     for i in range(map_reshaped.shape[3]):
# #         dst.write_band(i + 1, map_reshaped[:, :, :, i].astype(rasterio.float32))