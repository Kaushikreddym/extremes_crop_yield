import hydra
from omegaconf import DictConfig
from datasets.datasets import *
import importlib
from dask.distributed import Client, performance_report
import ipdb

@hydra.main(config_path="conf", config_name="config_datasets", version_base="1.3")
def main(cfg: DictConfig):
    # Set up Dask client (adapt based on your HPC environment)
    client = Client(n_workers=10, threads_per_worker=8, memory_limit="8GB")  # ~80 logical cores
    print(client)

    bounds = cfg.dataset.GDHY.bounds

    # Load primary datasets
    yield_data = load_GDHY(cfg.dataset.GDHY.path, bounds)
    crop_cal_data = load_SAGE(cfg.dataset.SAGE.path, bounds)
    crop_cal_data_50km = regrid(crop_cal_data, yield_data)

    # Load and regrid climate data
    climate_data = {}
    for name, attrs in cfg.dataset.MSWX.variables.items():
        ds = load_MSWX_zarr(attrs.path, name, attrs.var_name)
        ds_regrid = regrid(ds, yield_data)
        ds_mask = mask_crop_cal(ds_regrid, crop_cal_data_50km)

        # Ensure consistent units
        if attrs.var_name == 'precipitation':
            ds_mask.attrs['units'] = 'mm/day'
        elif attrs.var_name == 'air_temperature':
            ds_mask.attrs['units'] = 'degC'

        # Store single variable dataset
        climate_data[name] = ds_mask

    for index_cfg in cfg.indices.pr:
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
        result.to_zarr(zarr_filename, mode="w")

if __name__ == "__main__":
    main()
