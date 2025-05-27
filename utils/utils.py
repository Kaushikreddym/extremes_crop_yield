import xarray as xr
import xesmf as xe
import numpy as np

from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

def convert_longitude_to_minus180_180(ds):
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    ds = ds.sortby('lon')
    return ds
def regrid(ds_source,ds_target):
    regridder = xe.Regridder(ds_source, ds_target, method="bilinear")
    ds_regrid = regridder(ds_source)
    return ds_regrid
def mask_crop_cal(ds_source,ds_target):
    doy = ds_source.time.dt.dayofyear
    doy_3d = xr.DataArray(
        np.broadcast_to(doy.values[:, None, None], ds_source.shape),
        dims=['time', 'lat', 'lon'],
        coords={'time': ds_source['time'], 'lat': ds_source['lat'], 'lon': ds_source['lon']}
    )
    start_doy = ds_target['plant.start']
    end_doy = ds_target['harvest.end']
    start_3d = start_doy.expand_dims(time=ds_source.time)
    end_3d = end_doy.expand_dims(time=ds_source.time)
    crosses_year = start_3d > end_3d
    mask = xr.where(
        crosses_year,
        (doy_3d >= start_3d) | (doy_3d <= end_3d),
        (doy_3d >= start_3d) & (doy_3d <= end_3d)
    )
    return ds_source.where(mask)
def is_chunk_available(zarr_path, time_range=None, lat_range=None, lon_range=None):
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except Exception as e:
        print(f"Failed to open Zarr dataset: {e}")
        return False

    try:
        # Check time
        if time_range:
            if not ((ds.time.min() <= time_range[0]) and (ds.time.max() >= time_range[1])):
                return False

        # Check latitude
        if lat_range:
            if not ((ds.lat.min() <= lat_range[0]) and (ds.lat.max() >= lat_range[1])):
                return False

        # Check longitude
        if lon_range:
            if not ((ds.lon.min() <= lon_range[0]) and (ds.lon.max() >= lon_range[1])):
                return False

    except Exception as e:
        print(f"Error checking bounds: {e}")
        return False

    return True
