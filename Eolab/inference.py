# TODO use argparse and maybe even outsource it to its own class
# TODO convert to stand-alone script which can be run via the command line
# TODO needless data type conversion when saving output image?
# TODO package models into a library so that importing is easier
# TODO fix GPU inference: https://stackoverflow.com/questions/71278607/pytorch-expected-all-tensors-on-same-device
# TODO try inference with 500 pixel by 500 pixel size
import sys
from pathlib import Path
from re import search
from typing import List, Any, Union, Dict

import numpy as np
import rasterio
import rioxarray as rxr
import torch
import xarray
import geopandas as gpd
from shapely.geometry import mapping
import logging
from time import time

import models

logging.basicConfig(level=logging.INFO, filename="/home/eouser/git-repos/futureforest/Eolab/runtime.txt")

INPUT_TILE_PATH: str = "/home/eouser/git-repos/futureforest/Eolab/tiles.txt"
INPUT_WEIGHTS_PATH: str = "/home/eouser/git-repos/futureforest/Eolab/LSTM_26.pkl"
S2_BASE_DIR: Path = Path("/force/FORCE/C1/L2/ard/")
OUTPUT_BASE_DIR: Path = Path("/home/eouser/git-repos/futureforest/outputs/data")
MASK_BASE_DIR: Path = Path("/home/eouser/git-repos/futureforest/FORCE_minibatches")
LANDSHUT: str = "X0067_Y0058"
MINITILE: int = 31
CHUNK_SIZE = 1000  # GeoPackages are 500 pixel by 500 pixel
NEWEST_CUTOFF: int = 20230302

with open(INPUT_TILE_PATH, "rt") as f:
    FORCE_tiles: List[str] = [tile.replace("\n", "") for tile in f.readlines()]

lstm: torch.nn.LSTM = torch.load(INPUT_WEIGHTS_PATH, map_location=torch.device('cpu'))

template_metadata: Dict[str, Any] = {
    'driver': 'GTiff',
    'width': None,
    'height': None,
    'count': 1,
    'dtype': rasterio.float32,
    'crs': 'EPSG:3035',
    'transform': None
}


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
    outputs = model(data)
    _, predicted = torch.max(outputs.data, 1)
    return predicted


for tile in FORCE_tiles:
    start: float = time()
    #logging.info(f"Loading mask for FORCE tile {MASK_BASE_DIR / Path(LANDSHUT) / (str(MINITILE) + '.gpkg')}")
    #minitile = gpd.read_file(MASK_BASE_DIR / Path(LANDSHUT) / (str(MINITILE) + ".gpkg"))

    logging.info(f"Processing FORCE tile {tile}")
    s2_tile_dir: Path = S2_BASE_DIR / tile
    tile_paths: List[str] = [str(p) for p in s2_tile_dir.glob("*SEN*BOA.tif")]
    cube_inputs: List[str] = [
        tile_path for tile_path in tile_paths if int(search(r"\d{8}", tile_path).group(0)) <= NEWEST_CUTOFF
    ]

    # TODO there may exist an easier solution than opening an image from my tile stack
    with rasterio.open(cube_inputs[0]) as f:
        metadata = f.meta
        metadata["count"] = 1
        metadata["dtype"] = rasterio.float32

    assert metadata['height'] % 2 == 0 and metadata['width'] % 2 == 0

    tile_rows: int = metadata["height"]  # s2_cube_prediction.shape[2]
    tile_cols: int = metadata["width"]  # s2_cube_prediction.shape[3]
    output_torch: torch.tensor = torch.zeros([tile_rows, tile_cols])

    for row in range(0, tile_rows, CHUNK_SIZE):
        for col in range(0, tile_cols, CHUNK_SIZE):
            start_chunked: float = time()
            logging.info(f"Creating chunked data cube")
            s2_cube: Union[xarray.Dataset, xarray.DataArray, list[xarray.Dataset]] = []
            for cube_input in cube_inputs:
                layer: Union[xarray.Dataset, xarray.DataArray] = rxr.open_rasterio(cube_input)
                clipped_layer = layer.isel(x=slice(row, row + CHUNK_SIZE), y=slice(col, col + CHUNK_SIZE))
                s2_cube.append(clipped_layer)

            logging.info(f"Converting chunked data cube to numpy array")
            s2_cube_np: np.ndarray = np.array(s2_cube, ndmin=4)
            logging.info(f"Converting chunked numpy array to torch tensor")
            s2_cube_torch: torch.tensor = torch.tensor(s2_cube_np, dtype=torch.float32)
            logging.info(f"Permuting torch tensor")
            s2_cube_prediction: torch.tensor = s2_cube_torch.permute(2, 3, 0, 1)

            # TODO why not use smaller image tiles as input? is the lstm fixed to a certain input length?
            #  alternatively: below, the prediction is done row-wise per tile. The clipping should not matter in that
            #  case!
            for chunk_rows in range(0, CHUNK_SIZE):
                start_row: float = time()
                output_torch[row + chunk_rows, col:col + CHUNK_SIZE] = predict(lstm, s2_cube_prediction[chunk_rows])
                logging.info(f"Processed row {chunk_rows}/{CHUNK_SIZE - 1} of "
                             f"row-wise chunk: {row}:{row + CHUNK_SIZE - 1}, column-wise chunk: {col}:{col + CHUNK_SIZE - 1} "
                             f"({tile = }) in {time() - start_chunked:.2f} seconds")

            logging.info(f"Processed chunk in {time() - start_chunked} seconds")

    output_numpy: np.array = output_torch.numpy()

    # TODO save as int not float
    with rasterio.open(OUTPUT_BASE_DIR / (tile + ".tif"), 'w', **metadata) as dst:
        dst.write_band(1, output_numpy.astype(rasterio.float32))

    logging.info(f"Processed tile {tile} in {time() - start} seconds")

    del metadata
    del s2_cube
    del s2_cube_torch
    del s2_cube_prediction
    del output_torch
    del output_numpy

