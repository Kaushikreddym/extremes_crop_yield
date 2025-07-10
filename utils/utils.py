import xarray as xr
import xesmf as xe
import numpy as np

from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import os

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

from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import socket
def subset_by_bounds(ds, bounds, lat_name='lat', lon_name='lon'):
    return ds.sel(
        **{
            lat_name: slice(bounds['lat_max'], bounds['lat_min']),
            lon_name: slice(bounds['lon_min'], bounds['lon_max'])
        }
    )
def setup_slurm_cluster():
    cluster = SLURMCluster(
        queue="compute",  # or your SLURM partition name
        project="your_project",  # optional, depends on your HPC
        cores=8,  # number of cores per worker
        memory="16GB",  # memory per worker
        walltime="02:00:00",  # job walltime
        interface="ib0",  # or appropriate network interface (e.g., eth0, enp1s0)
        job_extra=[
            "--exclusive",  # optional: one job per node
            "--qos=normal"  # adjust QoS if needed
        ],
        job_script_prologue=[
            "module load anaconda",       # load necessary modules
            "source activate myenv",      # activate your conda/venv
        ],
        local_directory="/scratch/$USER/dask-workers",  # temp dir for dask workers
        log_directory="./dask_logs"  # local log folder
    )

    # Scale the cluster (e.g., 10 workers = 10 SLURM jobs)
    cluster.scale(jobs=10)

    # Attach a client to it
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)
    return client

def zarr_exists_with_bounds_and_time(zarr_path: str, time_range: dict, bounds: dict) -> bool:
    """
    Check if a Zarr store already exists with the specified time range and bounding box.
    """
    if not os.path.exists(zarr_path):
        return False

    try:
        print(f"loading the existing zarr store")
        ds = xr.open_zarr(zarr_path, consolidated=False)

        # Time range check
        start_time = str(ds.time[0].values)[:10]
        end_time = str(ds.time[-1].values)[:10]
        time_match = (
            start_time >= time_range['start_date'] and
            end_time <= time_range['end_date']
        )
        # Spatial bounds check
        buffer = 0.2
        lat = ds['lat']
        lon = ds['lon']
        lat_match = (
            float(lat.min()) >= bounds['lat_min'] - buffer and
            float(lat.max()) <= bounds['lat_max'] + buffer
        )
        lon_match = (
            float(lon.min()) >= bounds['lon_min'] - buffer and
            float(lon.max()) <= bounds['lon_max'] + buffer
        )
        return time_match and lat_match and lon_match

    except Exception as e:
        print(f"⚠️ Could not validate existing Zarr: {zarr_path}\n{e}")
        return False