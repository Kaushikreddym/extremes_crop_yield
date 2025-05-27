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

def load_MSWX_zarr(data_path,name,var_name):
    ds = xr.open_zarr(data_path)[var_name].chunk({'time': -1, 'lat': 50, 'lon': 50})
    return ds
