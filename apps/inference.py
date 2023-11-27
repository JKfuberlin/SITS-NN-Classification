# TODO package models into a library so that importing is easier -> requires re-training, if I'm not mistaken
# TODO fix GPU inference: https://stackoverflow.com/questions/71278607/pytorch-expected-all-tensors-on-same-device
# TODO try inference with 500 pixel by 500 pixel size
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

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-w", "--weights", dest="weights", required=True, type=Path)
parser.add_argument("--input-tiles", dest="input", required=True, type=Path)
parser.add_argument("--input-dir", dest="base", required=True, type=Path)
parser.add_argument("--output-dir", dest="out", required=True, type=Path)
parser.add_argument("--date-cutoff", dest="date", required=True, type=int)
parser.add_argument("--chunk-size", dest="chunk", required=False, type=int, default=1000)
parser.add_argument("--log", dest="log", required=False, action="store_true")
parser.add_argument("--log-file", dest="log-file", required=False, type=str)

cli_args: Dict[str, Union[Path, int, bool, str]] = vars(parser.parse_args())

if cli_args.get("log"):
    if cli_args.get("log_file"):
        logging.basicConfig(level=logging.INFO, filename=cli_args.get("log-file"))
    else:
        logging.basicConfig(level=logging.INFO)

with open(cli_args.get("input"), "rt") as f:
    FORCE_tiles: List[str] = [tile.replace("\n", "") for tile in f.readlines()]

lstm: torch.nn.LSTM = torch.load(cli_args.get("weights"), map_location=torch.device('cpu'))

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
    logging.info(f"Processing FORCE tile {tile}")
    s2_tile_dir: Path = cli_args.get("base") / tile
    tile_paths: List[str] = [str(p) for p in s2_tile_dir.glob("*SEN*BOA.tif")]
    cube_inputs: List[str] = [
        tile_path for tile_path in tile_paths if int(search(r"\d{8}", tile_path).group(0)) <= cli_args.get("date")
    ]

    with rasterio.open(cube_inputs[0]) as f:
        metadata = f.meta
        metadata["count"] = 1
        metadata["dtype"] = rasterio.uint8
        metadata["nodata"] = 0

    assert metadata['height'] % 2 == 0 and metadata['width'] % 2 == 0

    tile_rows: int = metadata["height"]  # s2_cube_prediction.shape[2]
    tile_cols: int = metadata["width"]  # s2_cube_prediction.shape[3]
    output_torch: torch.tensor = torch.zeros([tile_rows, tile_cols])

    for row in range(0, tile_rows, cli_args.get("chunk")):
        for col in range(0, tile_cols, cli_args.get("chunk")):
            start_chunked: float = time()
            logging.info(f"Creating chunked data cube")
            s2_cube: Union[xarray.Dataset, xarray.DataArray, list[xarray.Dataset]] = []
            for cube_input in cube_inputs:
                layer: Union[xarray.Dataset, xarray.DataArray] = rxr.open_rasterio(cube_input)
                clipped_layer = layer.isel(x=slice(row, row + cli_args.get("chunk")),
                                           y=slice(col, col + cli_args.get("chunk")))
                s2_cube.append(clipped_layer)

            logging.info(f"Converting chunked data cube to numpy array")
            s2_cube_np: np.ndarray = np.array(s2_cube, ndmin=4)
            logging.info(f"Converting chunked numpy array to torch tensor")
            s2_cube_torch: torch.tensor = torch.tensor(s2_cube_np, dtype=torch.float32)
            logging.info(f"Permuting torch tensor")
            s2_cube_prediction: torch.tensor = s2_cube_torch.permute(2, 3, 0, 1)

            for chunk_rows in range(0, cli_args.get("chunk")):
                start_row: float = time()
                output_torch[row + chunk_rows, col:col + cli_args.get("chunk")] = (
                    predict(lstm, s2_cube_prediction[chunk_rows]))
                logging.info(f"Processed row {chunk_rows}/{cli_args.get('chunk') - 1} of "
                             f"row-wise chunk: {row}:{row + cli_args.get('chunk') - 1}, "
                             f"column-wise chunk: {col}:{col + cli_args.get('chunk') - 1} "
                             f"({tile = }) in {time() - start_chunked:.2f} seconds")

            logging.info(f"Processed chunk in {time() - start_chunked} seconds")

    output_numpy: np.array = output_torch.numpy()

    with rasterio.open(cli_args.get("out") / (tile + ".tif"), 'w', **metadata) as dst:
        dst.write_band(1, output_numpy.astype(rasterio.uint8))

    logging.info(f"Processed tile {tile} in {time() - start} seconds")

    del metadata
    del s2_cube
    del s2_cube_torch
    del s2_cube_prediction
    del output_torch
    del output_numpy
