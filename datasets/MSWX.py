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
from datasets.datasets import *

warnings.filterwarnings("ignore", category=Warning)

class MSWX:
    def __init__(self, var_cfg: DictConfig):
        self.var_cfg = var_cfg
        self.files = []
        self.dataset = None

    def fetch(self):
        param_mapping = self.var_cfg.mappings
        provider = self.var_cfg.dataset.lower()
        parameter_key = self.var_cfg.weather.parameter

        param_info = param_mapping[provider]['variables'][parameter_key]
        folder_id = param_info["folder_id"]

        start_date = self.var_cfg.time_range.start_date
        end_date = self.var_cfg.time_range.end_date

        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        expected_files = []
        current = start
        while current <= end:
            doy = current.timetuple().tm_yday
            basename = f"{current.year}{doy:03d}.nc"
            expected_files.append(basename)
            current += timedelta(days=1)

        output_dir = self.var_cfg.data_dir
        provider = self.var_cfg.dataset.lower()
        parameter_key = self.var_cfg.weather.parameter
        local_files = []
        missing_files = []

        for basename in expected_files:
            local_path = os.path.join(output_dir, provider, parameter_key, basename)
            if os.path.exists(local_path):
                local_files.append(basename)
            else:
                missing_files.append(basename)

        if not missing_files:
            print(f"✅ All {len(expected_files)} files already exist locally. No download needed.")
            self.files = local_files
            return local_files

        print(f"📂 {len(local_files)} exist, {len(missing_files)} missing — fetching from Drive...")

        SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
        creds = service_account.Credentials.from_service_account_file(
            param_mapping[provider].params.google_service_account, scopes=SCOPES
        )
        service = build('drive', 'v3', credentials=creds)

        drive_files = list_drive_files(folder_id, service)
        valid_filenames = set(missing_files)
        files_to_download = [f for f in drive_files if f['name'] in valid_filenames]

        if not files_to_download:
            print(f"⚠️ None of the missing files found in Drive. Check folder & date range.")
            self.files = local_files
            return local_files

        for file in files_to_download:
            filename = file['name']
            local_path = os.path.join(output_dir, provider, parameter_key, filename)
            print(f"⬇️ Downloading {filename} ...")
            download_drive_file(file['id'], local_path, service)
            local_files.append(filename)

        self.files = local_files
        return local_files

    def load(self):
        param_mapping = self.var_cfg.mappings
        provider = self.var_cfg.dataset.lower()
        parameter_key = self.var_cfg.weather.parameter
        region = self.var_cfg.region
        bounds = self.var_cfg.bounds[region]

        param_info = param_mapping[provider]['variables'][parameter_key]
        output_dir = self.var_cfg.data_dir
        valid_dsets = []

        for f in self.files:
            local_path = os.path.join(output_dir, provider, parameter_key, f)
            try:
                ds = xr.open_dataset(local_path, chunks='auto', engine='netcdf4')[param_info.name]
                ds = subset_by_bounds(ds, bounds, lat_name='lat', lon_name='lon')
                valid_dsets.append(ds)
            except Exception as e:
                print(f"Skipping file due to error: {f}\n{e}")

        dset = xr.concat(valid_dsets, dim='time')
        dset = dset.transpose('time', 'lat', 'lon')
        self.dataset = dset
        return dset

    def to_zarr(self):
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call `load()` before `to_zarr()`.")

        var_name = self.var_cfg.weather.parameter
        dataset_name = self.var_cfg.dataset
        region = self.var_cfg.region

        # Add standard units metadata
        if var_name == 'pr':
            self.dataset.attrs['units'] = 'mm/day'
        elif var_name in ['tas', 'tasmax', 'tasmin']:
            self.dataset.attrs['units'] = 'degC'

        zarr_filename = self.var_cfg.output.filename.format(
            index=var_name,
            dataset=dataset_name,
            region=region,
            start=self.var_cfg.time_range.start_date,
            end=self.var_cfg.time_range.end_date,
            freq='1D',
        )
        zarr_path = os.path.join("data/MSWX/", zarr_filename)
        os.makedirs(os.path.dirname(zarr_path), exist_ok=True)

        print(f"💾 Saving {var_name} to Zarr: {zarr_path}")
        self.dataset.to_zarr(zarr_path, mode="w")
