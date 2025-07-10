import pandas as pd
import numpy as np
from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest
import geemap
import ee
import ipdb
import geopandas as gpd
from omegaconf import DictConfig
import os
import yaml
import time
from tqdm import tqdm
import warnings
from datetime import datetime, timedelta
import xarray as xr
import hydra
from omegaconf import DictConfig
import pint
import pint_pandas

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import io
import requests
from scipy.spatial import cKDTree
import argparse
import re

import requests
from bs4 import BeautifulSoup
import concurrent.futures

import gzip
from utils.utils import *

warnings.filterwarnings("ignore", category=Warning)

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

import os
from typing import Dict
import xarray as xr
from omegaconf import DictConfig
from glob import glob
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

import dask
from dask.diagnostics import ProgressBar
# from dask.distributed import Client, LocalCluster


# def MSWX_to_zarr(cfg: DictConfig):
#     warnings.filterwarnings("ignore")

#     europe_bounds = cfg.dataset.MSWX.bounds
#     mswx_path = cfg.dataset.MSWX.variables
#     output_dir = cfg.output_dir if "output_dir" in cfg else "."

#     def open_valid_datasets(file_list, chunks=None, engine='netcdf4'):
#         valid_dsets = []
#         for f in file_list:
#             try:
#                 ds = xr.open_dataset(f, chunks=chunks, engine=engine)
#                 valid_dsets.append(ds)
#             except Exception as e:
#                 print(f"Skipping file due to error: {f}\n{e}")
#         return valid_dsets

#     ys = int(cfg.dataset.time_range.start[:4])
#     ye = int(cfg.dataset.time_range.end[:4]) + 1
#     years = range(ys, ye)
#     chunking = {'time': 100, 'lat': 100, 'lon': 100}

#     def load_and_save(var_key, zarr_name):
#         var_cfg = mswx_path[var_key]
#         files = sorted(
#             f for year in years
#             for f in glob(os.path.join(var_cfg.path, f"{year}???.nc"))
#         )
#         dset = xr.concat(open_valid_datasets(files, chunks=chunking), dim='time')
#         dset = dset.transpose('time', 'lat', 'lon')
#         zarr_path = os.path.join(output_dir, f"{zarr_name}_{ys}-{ye}.zarr")
#         dset.chunk({'time': 50, 'lat': 50, 'lon': 50}).to_zarr(zarr_path, mode='w')
#         return zarr_path

#     paths = {}
#     paths["tasmin"] = load_and_save("tasmin", "tasmin_mswx_daily")
#     paths["tasmax"] = load_and_save("tasmax", "tasmax_mswx_daily")
#     paths["pr"]     = load_and_save("pr",     "pr_mswx_daily")
#     paths["tas"]     = load_and_save("tas",     "pr_mswx_daily")
#     # paths["hurs"]     = load_and_save("hurs",     "pr_mswx_daily")

#     return paths
# def load_MSWX_zarr(cfg: DictConfig, variable: str):
#     ys = int(cfg.dataset.time_range.start[:4])
#     ye = int(cfg.dataset.time_range.end[:4])
#     zarr_name = f"{variable}_mswx_daily_{ys}-{ye}.zarr"
#     zarr_path = os.path.join(cfg.output_dir, zarr_name)

#     if not os.path.exists(zarr_path):
#         print(f"Zarr file not found at {zarr_path}. Generating it...")
#         MSWX_to_zarr(cfg)
def list_drive_files(folder_id, service):
    """
    List all files in a Google Drive folder, handling pagination.
    """
    files = []
    page_token = None

    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id, name), nextPageToken",
            pageToken=page_token
        ).execute()

        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken", None)

        if not page_token:
            break

    return files
def download_drive_file(file_id, local_path, service):
    """
    Download a single file from Drive to a local path.
    """
    request = service.files().get_media(fileId=file_id)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with io.FileIO(local_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"   → Download {int(status.progress() * 100)}% complete")
# def fetch_MSWX(var_cfg):
#     param_mapping = var_cfg.mappings
#     provider = var_cfg.dataset.lower()
#     parameter_key = var_cfg.weather.parameter

#     param_info = param_mapping[provider]['variables'][parameter_key]
#     folder_id = param_info["folder_id"]

#     start_date = var_cfg.time_range.start_date
#     end_date = var_cfg.time_range.end_date

#     # === 1) Generate expected filenames ===
#     start = datetime.fromisoformat(start_date)
#     end = datetime.fromisoformat(end_date)

#     expected_files = []
#     current = start
#     while current <= end:
#         doy = current.timetuple().tm_yday
#         basename = f"{current.year}{doy:03d}.nc"
#         expected_files.append(basename)
#         current += timedelta(days=1)

#     output_dir = var_cfg.data_dir
#     local_files = []
#     missing_files = []

#     for basename in expected_files:
#         local_path = os.path.join(output_dir, provider, parameter_key, basename)
#         if os.path.exists(local_path):
#             local_files.append(basename)
#         else:
#             missing_files.append(basename)

#     if not missing_files:
#         print(f"✅ All {len(expected_files)} files already exist locally. No download needed.")
#         return local_files

#     print(f"📂 {len(local_files)} exist, {len(missing_files)} missing — fetching from Drive...")

#     # === 2) Connect to Drive ===
#     SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
#     creds = service_account.Credentials.from_service_account_file(
#         param_mapping[provider].params.google_service_account, scopes=SCOPES
#     )
#     service = build('drive', 'v3', credentials=creds)

#     # === 3) List all Drive files ===
#     drive_files = list_drive_files(folder_id, service)
#     valid_filenames = set(missing_files)

#     files_to_download = [f for f in drive_files if f['name'] in valid_filenames]

#     if not files_to_download:
#         print(f"⚠️ None of the missing files found in Drive. Check folder & date range.")
#         return local_files

#     # === 4) Download missing ===
#     for file in files_to_download:
#         filename = file['name']
#         local_path = os.path.join(output_dir, provider, parameter_key, filename)
#         print(f"⬇️ Downloading {filename} ...")
#         download_drive_file(file['id'], local_path, service)
#         local_files.append(filename)

#     return local_files

# def load_MSWX(var_cfg: DictConfig, files):
#     param_mapping = var_cfg.mappings
#     provider = var_cfg.dataset.lower()
#     parameter_key = var_cfg.weather.parameter
#     region = var_cfg.region
#     bounds = var_cfg.bounds[region]
    
#     param_info = param_mapping[provider]['variables'][parameter_key]
#     output_dir = var_cfg.data_dir
#     valid_dsets = []
#     for f in files:
#         local_path = os.path.join(output_dir, provider, parameter_key, f)
#         try:
#             ds = xr.open_dataset(local_path, chunks='auto', engine='netcdf4')[param_info.name]
#             ds = subset_by_bounds(ds, bounds, lat_name='lat', lon_name='lon')
#             valid_dsets.append(ds)
#         except Exception as e:
#             print(f"Skipping file due to error: {f}\n{e}")
#     dset = xr.concat(valid_dsets, dim='time')
#     dset = dset.transpose('time', 'lat', 'lon')
#     return dset

# def to_parquet(var_cfg: DictConfig):
#     yield_data = load_GDHY(cfg.mappings.GDHY.path, cfg.bounds[cfg.region])

#     ind='*'
#     region='europe'
#     files = glob(f'../data/10km/{ind}_mswx_{region}_1989-01-01_2024-12-31_YS_10km.zarr/')
#     indices = [f.split('/')[-2].split('_mswx_')[0] for f in files]

#     index_climate = []
#     for ind in indices:
#         print(f"Loading {ind} for {region}")
#         # Open the Zarr dataset
#         temp = xr.open_zarr(f'../data/10km/{ind}_mswx_{region}_1989-01-01_2024-12-31_YS_10km.zarr/')
#         first_var = list(temp.data_vars.keys())[0]
#         temp = temp.rename({first_var: ind})
#         # Append to the list
#         index_climate.append(temp)
        
#     index_climate = xr.merge(index_climate)

#     yield_data_10km = regrid(yield_data,index_climate)
#     yield_data_10km, index_climate = xr.align(yield_data_10km, index_climate)
#     df = xr.merge([index_climate,yield_data_10km]).to_dataframe().reset_index()
#     df.to_parquet(f'index_yield_mswx_{region}_1989-01-01_2024-12-31_YS_10km.parquet')

#     return df