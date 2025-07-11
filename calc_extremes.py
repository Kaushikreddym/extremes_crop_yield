from glob import glob
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
xr.set_options(keep_attrs=True)
import rioxarray
# import xarray.core.calendar as xcal
import cftime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import colormaps as cmaps

import seaborn as sns

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from datasets.datasets import *
from datasets.MSWX import *
from utils.utils import *
from datasets.indices import *
    
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    fetcher = MSWX(cfg)
    climate_data = {}
    for var in ['pr', 'tas', 'tasmax', 'tasmin']:
        cfg.weather.parameter = var

        zarr_filename = cfg.output.filename.format(
            index=var,
            dataset=cfg.dataset,
            region=cfg.region,
            start=cfg.time_range.start_date,
            end=cfg.time_range.end_date,
            freq='1D',
        )
        zarr_path = os.path.join("data/MSWX/", zarr_filename)
        time_range = {
            'start_date': cfg.time_range.start_date,
            'end_date': cfg.time_range.end_date
        }
        bounds = cfg.bounds[cfg.region]

        if zarr_exists_with_bounds_and_time(zarr_path, time_range, bounds):
            print(f"✅ Skipping {var} — already exists with matching bounds and time.")
            var_name=cfg.mappings[cfg.dataset].variables[var].name
            ds = xr.open_zarr(zarr_path)
            climate_data[var] = ds[var_name].chunk({'lat': 10, 'lon': 10, 'time': -1})
            continue

        fetcher.fetch()
        fetcher.load()
        fetcher.to_zarr()
        # After saving, load the dataset
        var_name=cfg.mappings[cfg.dataset].variables[var].name
        ds = xr.open_zarr(zarr_path)
        climate_data[var] = ds[var_name].chunk({'lat': 10, 'lon': 10, 'time': -1})
        
        
    indices = extreme_index(cfg, climate_data)
    # indices.calculate('tn10p')
    indices.run()
if __name__ == "__main__":
    from dask.distributed import Client, performance_report
    client = Client(n_workers=20, threads_per_worker=2, memory_limit="80GB")  # ~80 logical cores

    import multiprocessing
    multiprocessing.freeze_support()
    main()