import hydra
from omegaconf import DictConfig
from datasets.datasets import *
import ipdb
import importlib

@hydra.main(config_path="conf", config_name="config_datasets", version_base="1.3")
def main(cfg: DictConfig):    
    bounds = cfg.dataset.GDHY.bounds

    data_path = cfg.dataset.GDHY.path
    yield_data = load_GDHY(data_path, bounds)
    
    data_path = cfg.dataset.SAGE.path
    crop_cal_data = load_SAGE(data_path, bounds)
    crop_cal_data_50km = regrid(crop_cal_data, yield_data)

    climate_data = []
    config = cfg.dataset.MSWX
    climate_data = {}
    for name, attrs in config.variables.items():
        # ipdb.set_trace()
        ds = load_MSWX_zarr(attrs.path, name, attrs.var_name)
        ds_regrid = regrid(ds, yield_data)
        ds_mask = mask_crop_cal(ds_regrid, crop_cal_data_50km)

        if attrs.var_name == 'precipitation':
            ds_mask.attrs['units'] = 'mm/day'
        elif attrs.var_name == 'air_temperature':
            ds_mask.attrs['units'] = 'degC'
        ds_mask = ds_mask.to_dataset(name=name)
        climate_data[name] = ds_mask
    # climate_data = xr.merge(climate_data)
    # print(f"Loaded variables: {list(climate_data.keys())}")
    # 
    
    derived = {"pr": climate_data['pr']}
    for index_cfg in cfg.indices.pr:
        name = index_cfg.name
        func_path = index_cfg.function
        args = index_cfg.args
        
        if "preprocess" in index_cfg:
            for group, variables in index_cfg.preprocess.items():
                for var_name, task in variables.items():
                    input_var = derived[task["input"]]
                    op = getattr(input_var, task["operation"])
                    result = op(**task.get("kwargs", {}))
                    derived[var_name] = result

        # Resolve any references in args (e.g., ${quantiles.p75})
        for key, val in args.items():
            if isinstance(val, str) and val.startswith("${"):
                ref = val.strip("${}").split(".")[-1]
                args[key] = derived[ref]

        module_name, func_name = func_path.rsplit(".", 1)
        func = getattr(importlib.import_module(module_name), func_name)
        result = func(climate_data['pr'], **args)
        
        zarr_filename = cfg.output.zarr_pattern.format(
            name=name,
            start=cfg.output.start,
            end=cfg.output.end,
            freq=args["freq"]
        )
        # add commit
        result.to_zarr(zarr_filename, mode="w")
if __name__ == "__main__":
    main()
