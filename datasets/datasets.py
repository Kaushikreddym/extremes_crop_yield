import os
import re
import xarray as xr
import pandas as pd
from glob import glob
from utils.utils import *
import gzip
import io

def get_yield_data(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'(\d{4})', filename)
    if not match:
        raise ValueError("No 4-digit year found in filename.")
    year = int(match.group(1))
    time = pd.to_datetime(f"{year}-01-01")
    ds = xr.open_dataset(filepath)
    ds = ds.expand_dims(time=[time])
    ds = ds.assign_coords(time=("time", [time]))
    return ds

def load_GDHY(data_path, bounds):

    GDHY_crops_path = [d for d in glob(os.path.join(data_path, '*')) if os.path.isdir(d)]
    crop_dset = []

    for GDHY_path in GDHY_crops_path:
        crop_name = os.path.basename(GDHY_path)
        print(f"Loading {crop_name} Yield Data from GDHY")
        GDHY_files = sorted(glob(os.path.join(GDHY_path, '*.nc4')))
        yield_dset = convert_longitude_to_minus180_180(
            xr.concat([get_yield_data(f) for f in GDHY_files], dim='time')
        )
        yield_dset = yield_dset.rename({'var': crop_name})
        yield_dset = yield_dset.sel(
            lat=slice(bounds.lat_min, bounds.lat_max),
            lon=slice(bounds.lon_min, bounds.lon_max)
        )
        crop_dset.append(yield_dset)

    return xr.merge(crop_dset)

def load_SAGE(data_path, bounds):
    compressed_paths = glob(data_path)
    with gzip.open(compressed_paths[0], 'rb') as f:
        uncompressed_data = f.read()

    crop_cal_SAGE = xr.open_dataset(io.BytesIO(uncompressed_data))
    crop_cal_SAGE = crop_cal_SAGE.rename({'longitude': 'lon', 'latitude': 'lat'})
    crop_cal_SAGE = crop_cal_SAGE.sel(
        lat=slice(bounds["lat_max"], bounds["lat_min"]),
        lon=slice(bounds["lon_min"], bounds["lon_max"])
    )
    return crop_cal_SAGE
def MSWX_to_zarr():
    from glob import glob
    import os

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import xarray as xr

    import warnings
    warnings.filterwarnings("ignore")

    from dask.diagnostics import ProgressBar
    from dask.distributed import Client, LocalCluster

    # -----------------------------
    # Setup Dask (optional but recommended)
    # -----------------------------
    cluster = LocalCluster(n_workers=80, threads_per_worker=1, memory_limit='80GB')
    client = Client(cluster)

    europe_bounds = {
        "lat_min": 34.5,
        "lat_max": 71.2,
        "lon_min": -25.0,
        "lon_max": 30.0
    }
    data01_path = '/data01/FDS/muduchuru'
    beegfs_path = '/beegfs/muduchuru/data'
    imerg_path = f'{data01_path}/IMERG/precip_data/'
    mswx_path = f'{beegfs_path}/MSWX_NC/1D/'
    import xesmf as xe

    def open_valid_datasets(file_list, chunks=None, engine='netcdf4'):
        valid_dsets = []
        for f in file_list:
            try:
                ds = xr.open_dataset(f, chunks=chunks, engine=engine)
                valid_dsets.append(ds)
            except Exception as e:
                print(f"Skipping file due to error: {f}\n{e}")
        return valid_dsets
    # -----------------------------
    # Define paths and years
    # -----------------------------
    ys = 1989
    ye= 2020
    years = range(ys, ye)  # Inclusive of 2010
    chunking = {'time': 100, 'lat': 100, 'lon': 100}

    # -----------------------------
    # Load MSWEP Precipitation
    # -----------------------------
    mswep_files = sorted(
        f for year in years
        for f in glob(f"{mswx_path}/pr/{year}???.nc")
    )
    pr_dset = xr.concat(
        open_valid_datasets(mswep_files, chunks=chunking),
        dim='time'
    )

    # -----------------------------
    # Load Tasmax and Tasmin (if needed)
    # -----------------------------
    tasmax_files = sorted(
        f for year in years
        for f in glob(f"{mswx_path}/tasmax/{year}???.nc")
    )
    tasmax_dset = xr.concat(
        open_valid_datasets(tasmax_files, chunks=chunking),
        dim='time'
    ).transpose('time', 'lat', 'lon')

    tasmin_files = sorted(
        f for year in years
        for f in glob(f"{mswx_path}/tasmin/{year}???.nc")
    )
    tasmin_dset = xr.concat(
        open_valid_datasets(tasmin_files, chunks=chunking),
        dim='time'
    ).transpose('time', 'lat', 'lon')

    tasmin_dset.chunk({'time': 50, 'lat': 50, 'lon': 50}).to_zarr(f"tasmin_mswx_daily_{ys}-{ye}.zarr",mode='w')
    tasmax_dset.chunk({'time': 50, 'lat': 50, 'lon': 50}).to_zarr(f"tasmax_mswx_daily_{ys}-{ye}.zarr",mode='w')
    pr_dset.chunk({'time': 50, 'lat': 50, 'lon': 50}).to_zarr(f"pr_mswep_daily_{ys}-{ye}.zarr",mode='w')

def load_MSWX_zarr(data_path,name,var_name):
    ds = xr.open_zarr(data_path)[var_name].chunk({'time': -1, 'lat': 50, 'lon': 50})
    return ds
