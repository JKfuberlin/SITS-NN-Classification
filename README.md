# TSC_CNN
Convolutional neural network for tree species classification

## Installation

### Python Wheels

**Currently not available!**

### From Repository

#### Pytorch

Unfortunately, the dependency management poetry offers makes the installation of pytorch somewhat cumbersome. By default,
the CUDA 12.1 versions of Pytorch installed. Should you want to install other versions (i.e. CPU wheels or CUDA 11.8),
the following commands are necessary after installation:

```bash
poetry install

# for CUDA 11.8
poetry remove torch torchvision torchaudio
poetry add --source=pytorch_cu118 torch torchvision torchaudio

# for CPU wheels
poetry remove torch torchvision torchaudio
poetry add --source=pytorch_cpu torch torchvision torchaudio
```

To revert back to the CUDA 12.1 wheels, run:

```bash
poetry remove torch torchvision torchaudio
poetry add torch torchvision torchaudio
```
## Usage

### Standalone Scripts

Tree species can be predicted with the standalone `inference.py` script. Currently, inference is possible with **LSTM 
classifier only**. Please note, that a [FORCE](https://force-eo.readthedocs.io/en/latest/) datacube is expected as input.
If you installed `sits_classifier` by cloning this repository and running `poetry install`, you must work within the 
poetry shell which masks the python interpreter. All other installed system binaries are still available to you. 

> Note that other environment managers such as conda should probably be quit beforehand. Thus, running e.g. 
> `conda deactivate` is suggested. 

```bash
poetry shell

python apps/inference.py --help

# usage: inference.py [-h] -w WEIGHTS --input-tiles INPUT --input-dir BASE --output-dir OUT --date-cutoff DATE [--chunk-size CHUNK] [--log] [--log-file LOG-FILE]
# 
# Run inference with already trained LSTM classifier on a remote-sensing time series represented as FORCE ARD datacube.
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   -w WEIGHTS, --weights WEIGHTS
#                         Path to pre-trained classifier to be loaded via `torch.load`. Can be either a relative or absolute file path.
#   --input-tiles INPUT   List of FORCE tiles which should be used for inference. Each line should contain one FORCE tile specifier (Xdddd_Ydddd).
#   --input-dir BASE      Path to FORCE datacube.
#   --output-dir OUT      Path to directory into which predictions should be saved.
#   --date-cutoff DATE    Cutoff date for time series which should be included in datacube for inference.
#   --chunk-size CHUNK    Chunk size which further subsets FORCE tiles for RAM-friendly prediction.
#   --log                 Emit logs?
#   --log-file LOG-FILE   If logging is enabled, write to this file. If omitted, logs are written to stdout.
```