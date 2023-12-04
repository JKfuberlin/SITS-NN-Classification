# TODO fix GPU inference: https://stackoverflow.com/questions/71278607/pytorch-expected-all-tensors-on-same-device
# TODO masking strategy: mask each scene or the entire datacube? Or both?
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
parser.add_argument("--output-dir", dest="out", required=True, type=Path,
                    help="Path to directory into which predictions should be saved.")
parser.add_argument("--date-cutoff", dest="date", required=True, type=int,
                    help="Cutoff date for time series which should be included in datacube for inference.")
parser.add_argument("--chunk-size", dest="chunk", required=False, type=int, default=1000,
                    help="Chunk size which further subsets FORCE tiles for RAM-friendly prediction.")
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
    sorted(tile_paths, key=lambda x: int(x.split("/")[-1].split("_")[0]))
    cube_inputs: List[str] = [
        tile_path for tile_path in tile_paths if int(search(r"\d{8}", tile_path).group(0)) <= cli_args.get("date")
    ]

    with rasterio.open(cube_inputs[0]) as f:
        metadata = f.meta
        metadata["count"] = 1
        metadata["dtype"] = rasterio.uint8
        metadata["nodata"] = 0
        row_block, col_block = f.block_shapes[0]

    tile_rows: int = metadata["height"]  # s2_cube_prediction.shape[2]
    tile_cols: int = metadata["width"]  # s2_cube_prediction.shape[3]
    output_torch: torch.tensor = torch.zeros([tile_rows, tile_cols])

    # TODO instead of cli_args.get("chunk"), check what the user wants.
    #  If the block sizes should be used, substitute the step size.
    #  If only one of the two CL args are given, I need a different loops depending on what's selected.
    #    If one argument is missing, use tile_* as the step size and argument in layer.isel(). The code
    #    does not need to be further split/abstracted into functions if I'm not mistaken.
    row_step: int = cli_args.get("row-block") or row_block
    col_step: int = cli_args.get("col_block") or col_block

    if tile_rows % row_step != 0 or tile_cols % col_step != 0:
        raise AssertionError("Rows and columns must be divisible by their respective step sizes without remainder.")

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
            del s2_cube
            logging.info(f"Converting chunked numpy array to torch tensor")
            s2_cube_torch: torch.tensor = torch.as_tensor(s2_cube_np)
            logging.info(f"Permuting torch tensor")
            s2_cube_prediction: torch.tensor = s2_cube_torch.permute(2, 3, 0, 1)

            for chunk_rows in range(0, row_step):
                start_row: float = time()
                output_torch[row + chunk_rows, col:col + cli_args.get("chunk")] = (
                    predict(lstm, s2_cube_prediction[chunk_rows]))
                # TODO  FIXME the log message below is not up to date with the dataset
                logging.info(f"Processed row {chunk_rows}/{cli_args.get('chunk') - 1} of "
                             f"row-wise chunk: {row}:{row + cli_args.get('chunk') - 1}, "
                             f"column-wise chunk: {col}:{col + cli_args.get('chunk') - 1} "
                             f"({tile = }) in {time() - start_row:.2f} seconds")

            logging.info(f"Processed chunk in {time() - start_chunked} seconds")

            del s2_cube_np
            del s2_cube_torch
            del s2_cube_prediction

    output_numpy: np.array = output_torch.numpy()

    with rasterio.open(cli_args.get("out") / (tile + ".tif"), 'w', **metadata) as dst:
        dst.write_band(1, output_numpy.astype(rasterio.uint8))

    logging.info(f"Processed tile {tile} in {time() - start} seconds")

    del metadata
    del output_torch
    del output_numpy
