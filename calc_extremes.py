import hydra
from omegaconf import DictConfig
from datasets.datasets import *
from utils.utils import *
import importlib
from dask.distributed import Client, performance_report
import ipdb

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Set up Dask client (adapt based on your HPC environment)
    client = Client(n_workers=40, threads_per_worker=2, memory_limit="80GB")  # ~80 logical cores

    bounds = cfg.bounds
    # Load primary datasets
    # yield_data = load_GDHY(cfg.dataset.GDHY.path, bounds)
    crop_cal_data = load_SAGE(cfg.mappings.SAGE.path, bounds)
    # crop_cal_data_50km = regrid(crop_cal_data, yield_data)

    # Load and regrid climate data
    climate_data = {}
    for name in ['tasmax']:
        cfg.weather.parameter = name
        files = fetch_MSWX(cfg)
        dset_weather = load_MSWX(cfg,files)
        # ds_regrid = regrid(ds, yield_data)
        # yield_regrid = regrid(yield_data,ds)
        crop_cal_data_10km = regrid(crop_cal_data, dset_weather)

        ds_mask = mask_crop_cal(dset_weather, crop_cal_data_10km)

        # Ensure consistent units
        if name == 'pr':
            ds_mask.attrs['units'] = 'mm/day'
        elif name in ['tasmax', 'tasmin', 'tas']:
            ds_mask.attrs['units'] = 'degC'

        # Store single variable dataset
        climate_data[name] = ds_mask

    for index_cfg in cfg.mappings.indices:
        name = index_cfg.name
        func_path = index_cfg.function
        args = dict(index_cfg.args)

        # Handle preprocessing
        if "preprocess" in index_cfg:
            for _, variables in index_cfg.preprocess.items():
                for var_name, task in variables.items():
                    input_var = climate_data[task["input"]]
                    op = getattr(input_var, task["operation"])
                    result = op(**task.get("kwargs", {}))
                    # Add aggregation if specified
                    agg_func = task.get("aggregation", None)
                    if agg_func:
                        result = getattr(result, agg_func)()
                    climate_data[var_name] = result

        # Resolve references in args
        for key, val in args.items():
            if isinstance(val, str) and val.startswith("${"):
                ref = val.strip("${}").split(".")[-1]
                args[key] = climate_data[ref]

        # Load and call function
        module_name, func_name = func_path.rsplit(".", 1)
        func = getattr(importlib.import_module(module_name), func_name)

        # Multiple variable support
        if hasattr(index_cfg, "variables"):
            ipdb.set_trace()
            inputs = [climate_data[v] for v in index_cfg.variables]
            result = func(*inputs, **args)
        else:
            result = func(climate_data["pr"], **args)
        # ipdb.set_trace()
        # Save result to zarr
        freq = args.get("freq", "YS")
        zarr_filename = cfg.output.zarr_pattern.format(
            name=name,
            start=cfg.output.start,
            end=cfg.output.end,
            freq=freq
        )
        os.makedirs("data/10km/",exist_ok=True)
        result.to_zarr("data/10km/"+zarr_filename, mode="w")

if __name__ == "__main__":
    main()
