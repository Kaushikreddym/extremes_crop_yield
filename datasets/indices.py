import os
import importlib
from copy import deepcopy
from pathlib import Path
import xarray as xr

class extreme_index:
    def __init__(self, cfg, climate_data):
        self.cfg = cfg
        self.climate_data = climate_data
        self.successful_indices = []
        self.failed_indices = []
        self.tasks = []

    def zarr_output_path(self, index):
        zarr_filename = self.cfg.output.filename.format(
            index=index,
            dataset=self.cfg.dataset,
            region=self.cfg.region,
            start=self.cfg.time_range.start_date,
            end=self.cfg.time_range.end_date,
            freq='YS',
        )
        return Path('data/10km/' + zarr_filename)

    def calculate(self, index):
        index_cfg = self.cfg.mappings.indices[index]
        # local_climate_data = deepcopy(self.climate_data)
        args = dict(index_cfg.args)

        # Handle preprocessing
        if "preprocess" in index_cfg:
            for _, variables in index_cfg.preprocess.items():
                for var_name, task in variables.items():
                    input_var = self.climate_data[task["input"]]
                    op = getattr(input_var, task["operation"])
                    result = op(**task.get("kwargs", {}))
                    if "aggregation" in task:
                        result = getattr(result, task["aggregation"])()
                    self.climate_data[var_name] = result

        # Resolve references in args
        for key, val in args.items():
            if isinstance(val, str) and val.startswith("${"):
                ref = val.strip("${}").split(".")[-1]
                args[key] = self.climate_data[ref]

        # Load function
        module_name, func_name = index_cfg.function.rsplit(".", 1)
        func = getattr(importlib.import_module(module_name), func_name)

        # Call function
        if hasattr(index_cfg, "variables"):
            inputs = [self.climate_data[v] for v in index_cfg.variables]
            result = func(*inputs, **args)
        else:
            result = func(self.climate_data["pr"], **args)

        # Save result
        zarr_path = self.zarr_output_path(index)
        os.makedirs(zarr_path.parent, exist_ok=True)
        result.to_zarr(str(zarr_path), mode="w")

    def run(self):
        for index in list(cfg.mappings.indices.keys()):
            zarr_path = self.zarr_output_path(index)
            if zarr_path.exists():
                print(f"Skipping {index.name}: Zarr already exists at {zarr_path}")
                continue
            try:
                print(f"Processing index: {index}")
                self.calculate(index)
                self.successful_indices.append(index)
            except Exception as e:
                print(f"Error processing {index}: {e}")
                self.failed_indices.append(index)