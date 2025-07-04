import hydra
from omegaconf import DictConfig
from datasets.datasets import *
from utils.utils import *
import importlib
# from dask.distributed import Client, performance_report
import ipdb
from dask import delayed

# @delayed
def process_index(index_cfg, climate_data, cfg):
    import importlib
    import os
    from copy import deepcopy

    # Deepcopy to avoid shared state
    local_climate_data = deepcopy(climate_data)
    args = dict(index_cfg.args)

    # Handle preprocessing
    if "preprocess" in index_cfg:
        for _, variables in index_cfg.preprocess.items():
            for var_name, task in variables.items():
                input_var = local_climate_data[task["input"]]
                op = getattr(input_var, task["operation"])
                result = op(**task.get("kwargs", {}))
                # Add aggregation if specified
                agg_func = task.get("aggregation", None)
                if agg_func:
                    result = getattr(result, agg_func)()
                local_climate_data[var_name] = result

    # Resolve references in args
    for key, val in args.items():
        if isinstance(val, str) and val.startswith("${"):
            ref = val.strip("${}").split(".")[-1]
            args[key] = local_climate_data[ref]

    # Load and call function
    module_name, func_name = index_cfg.function.rsplit(".", 1)
    func = getattr(importlib.import_module(module_name), func_name)
    
    # Support multiple input variables
    if hasattr(index_cfg, "variables"):
        inputs = [local_climate_data[v] for v in index_cfg.variables]
        result = func(*inputs, **args)
    else:
        result = func(local_climate_data["pr"], **args)

    # Save result to zarr
    freq = args.get("freq", "YS")
    zarr_filename = cfg.output.filename.format(
        index=index_cfg.name,
        dataset=cfg.dataset,
        region=cfg.region,
        start=cfg.time_range.start_date,
        end=cfg.time_range.end_date,
        freq=freq,
    )
    os.makedirs("data/10km/", exist_ok=True)
    result.to_zarr("data/10km/" + zarr_filename, mode="w")

def climate_to_zarr(cfg: DictConfig):
    region = cfg.region
    bounds = cfg.bounds[region]
    # Load primary datasets
    # yield_data = load_GDHY(cfg.dataset.GDHY.path, bounds)
    crop_cal_data = load_SAGE(cfg.mappings.SAGE.path, bounds)
    # crop_cal_data_50km = regrid(crop_cal_data, yield_data)

    # Load and regrid climate data
    # climate_data = {}
    for name in ['pr','tas','tasmax','tasmin']:
        cfg.weather.parameter = name
        files = fetch_MSWX(cfg)
        dset_weather = load_MSWX(cfg,files)
        # ds_regrid = regrid(ds, yield_data)
        # yield_regrid = regrid(yield_data,ds)
        crop_cal_data_10km = regrid(crop_cal_data, dset_weather)

        # ds_mask = mask_crop_cal(dset_weather, crop_cal_data_10km)
        ds_mask = dset_weather
        # Ensure consistent units
        if name == 'pr':
            ds_mask.attrs['units'] = 'mm/day'
        elif name in ['tasmax', 'tasmin', 'tas']:
            ds_mask.attrs['units'] = 'degC'

        # Store single variable dataset
        zarr_filename = cfg.output.filename.format(
            index=name,
            dataset=cfg.dataset,
            region=cfg.region,
            start=cfg.time_range.start_date,
            end=cfg.time_range.end_date,
            freq='1D',
        )
        os.makedirs("data/MSWX/", exist_ok=True)
        ds_mask.to_zarr("data/MSWX/" + zarr_filename, mode="w")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Set up Dask client (adapt based on your HPC environment)
    # client = Client(n_workers=20, threads_per_worker=2, memory_limit="80GB")  # ~80 logical cores
    
    # climate_to_zarr(cfg)
    climate_data = {}
    for name in ['pr','tas','tasmax','tasmin']:
        zarr_filename = cfg.output.filename.format(
            index=name,
            dataset=cfg.dataset,
            region=cfg.region,
            start=cfg.time_range.start_date,
            end=cfg.time_range.end_date,
            freq='1D',
        )
        os.makedirs("data/MSWX/", exist_ok=True)
        var_name=cfg.mappings[cfg.dataset].variables[name].name
        climate_data[name]=xr.open_zarr("data/MSWX/" + zarr_filename)[var_name].chunk({'lat': 10, 'lon': 10, 'time': -1})
        successful_indices = []
        failed_indices = []
        tasks = []
        


        for index_cfg in cfg.mappings.indices:
            zarr_filename = cfg.output.filename.format(
                index=index_cfg.name,
                dataset=cfg.dataset,
                region=cfg.region,
                start=cfg.time_range.start_date,
                end=cfg.time_range.end_date,
                freq='YS',
            )
            from pathlib import Path
            zarr_path = Path('data/10km/'+zarr_filename)
            if zarr_path.exists():
                print(f"Skipping {index_cfg.name}: Zarr already exists at {zarr_path}")
                continue
            try:
                print(f"Processing index: {index_cfg.name}")  # <-- Print current index
                result = process_index(index_cfg, climate_data, cfg)
                tasks.append(result)
                successful_indices.append(index_cfg.name)
            except Exception as e:
                print(f"Error processing {index_cfg.name}: {e}")
                failed_indices.append(index_cfg.name)


if __name__ == "__main__":
    main()