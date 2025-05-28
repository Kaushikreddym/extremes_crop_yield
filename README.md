# Climate Data Processing Pipeline

This repository contains a climate data processing pipeline using Hydra for configuration management. The main script (`main.py`) loads agricultural yield and crop calendar data, processes climate variables, computes derived climate indices, and stores the results in Zarr format.

## Features

- Loads GDHY yield and SAGE crop calendar data
- Regrids crop calendar and climate data to match yield data resolution
- Loads and processes MSWX climate data from Zarr stores
- Applies preprocessing steps and computes custom climate indices
- Outputs results in Zarr format using user-defined filename patterns

## Requirements

- Python 3.8+
- [Hydra](https://hydra.cc/)
- [xarray](https://docs.xarray.dev/)
- [zarr](https://zarr.readthedocs.io/)
- [omegaconf](https://omegaconf.readthedocs.io/)
- [ipdb](https://pypi.org/project/ipdb/)
- Custom dataset loaders: `load_GDHY`, `load_SAGE`, `load_MSWX_zarr`
- Regridding and masking utilities: `regrid`, `mask_crop_cal`

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
