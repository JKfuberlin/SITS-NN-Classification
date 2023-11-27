# TSC_CNN
Convolutional neural network for tree species classification

## Installation

### Python Wheels

### From Repository

#### Pytorch

Unfortunately, the dependency management poetry offers makes the installation of pytorch somewhat cumbersome. By default,
the CPU-versions of Pytorch are listed as dependencies. Thus, running `poetry install` will install the CPU-wheels.
Should the CUDA-wheels be installed, the following commands are necessary after installation:

```bash
poetry install

# for CUDA 11.8
poetry remove torch torchvision torchaudio
poetry add --source=pytorch_cu118 torch torchvision torchaudio

# for CUDA 12.1
poetry remove torch torchvision torchaudio
poetry add torch torchvision torchaudio
```

To revert back to the CPU-wheels, run:

```bash
poetry remove torch torchvision torchaudio
poetry add --source=pytorch_cpu torch torchvision torchaudio
```

## Usage

### Standalone Scripts