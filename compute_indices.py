import hydra
from omegaconf import DictConfig
from datasets.datasets import *
import importlib
from dask.distributed import Client

@hydra.main(config_path="conf", config_name="config_datasets", version_base="1.3")
def main(cfg: DictConfig):

    client = Client(n_workers=40, threads_per_worker=2, memory_limit="80GB")
    print(client)

    bounds = cfg.dataset.GDHY.bounds
    yield_data = load_GDHY(cfg.dataset.GDHY.path, bounds)
    crop_cal_data = load_SAGE(cfg.dataset.SAGE.path, bounds)
    crop_cal_data_50km = regrid(crop_cal_data, yield_data)

    climate_data = {}
    for name, attrs in cfg.dataset.MSWX.variables.items():
        ds = load_MSWX_zarr(attrs.path, name, attrs.var_name)
        ds_regrid = regrid(ds, yield_data)
        ds_mask = mask_crop_cal(ds_regrid, crop_cal_data_50km)
        ds_mask.attrs['units'] = 'mm/day' if attrs.var_name == 'precipitation' else 'degC'
        climate_data[name] = ds_mask

    for category in cfg.indices:
        for index_cfg in cfg.indices[category]:
            if index_cfg.name != cfg.index_name:
                continue

            name = index_cfg.name
            func_path = index_cfg.function
            args = dict(index_cfg.args)

            if "preprocess" in index_cfg:
                for _, variables in index_cfg.preprocess.items():
                    for var_name, task in variables.items():
                        input_var = climate_data[task["input"]]
                        op = getattr(input_var, task["operation"])
                        result = op(**task.get("kwargs", {}))
                        if agg := task.get("aggregation", None):
                            result = getattr(result, agg)()
                        climate_data[var_name] = result

            for key, val in args.items():
                if isinstance(val, str) and val.startswith("${"):
                    ref = val.strip("${}").split(".")[-1]
                    args[key] = climate_data[ref]

            module_name, func_name = func_path.rsplit(".", 1)
            func = getattr(importlib.import_module(module_name), func_name)
            inputs = [climate_data[v] for v in index_cfg.get("variables", ["pr"])]
            result = func(*inputs, **args)

            freq = args.get("freq", "YS")
            zarr_filename = cfg.output.zarr_pattern.format(
                name=name,
                start=cfg.output.start,
                end=cfg.output.end,
                freq=freq
            )
            result.to_zarr(zarr_filename, mode="w")
            return

if __name__ == "__main__":
    main()
