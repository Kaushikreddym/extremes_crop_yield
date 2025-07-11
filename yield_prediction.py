import os
import sys
import argparse
import warnings
from glob import glob

import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from xgboost import XGBRegressor

import optuna
from optuna.samplers import TPESampler
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess

import geopandas as gpd
import xarray as xr

from datasets.datasets import *
from datasets.indices import *
from utils.utils import *
from utils.ml import *

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = 'Times New Roman'

def combine_yield_and_indices(cfg, parquet_path):
    yield_data = load_GDHY(cfg.mappings.GDHY.path, cfg.bounds[cfg.region])

    ind='*'
    region = cfg.region
    files = glob(f'data/10km/{ind}_mswx_{region}_1989-01-01_2024-12-31_YS_10km.zarr/')
    indices = [f.split('/')[-2].split('_mswx_')[0] for f in files]

    index_climate = []
    for ind in indices:
        print(f"Loading {ind} for {region}")
        # Open the Zarr dataset
        temp = xr.open_zarr(f'data/10km/{ind}_mswx_{region}_1989-01-01_2024-12-31_YS_10km.zarr/')
        first_var = list(temp.data_vars.keys())[0]
        temp = temp.rename({first_var: ind})
        # Append to the list
        index_climate.append(temp)
        
    index_climate = xr.merge(index_climate)
    # import ipdb; ipdb.set_trace()
    yield_data_10km = regrid(yield_data,index_climate)
    yield_data_10km, index_climate = xr.align(yield_data_10km, index_climate)
    df = xr.merge([index_climate,yield_data_10km]).to_dataframe().reset_index()
    df.to_parquet(parquet_path)
    return indices

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    parquet_filename = cfg.output.filename_index.format(
        index='index_yield',
        dataset=cfg.dataset,
        region=cfg.region,
        start=cfg.time_range.start_date,
        end=cfg.time_range.end_date,
        freq='YS',
            )
    parquet_path = Path("data/combined/") / parquet_filename
    if not parquet_path.exists():
        print(f"Loading | file does not exist at {parquet_path}")
        indices = combine_yield_and_indices(cfg, parquet_path)
    else:
        # Get indices from file names
        region = cfg.region
        files = glob(f'data/10km/*_mswx_{region}_1989-01-01_2024-12-31_YS_10km.zarr/')
        indices = [f.split('/')[-2].split('_mswx_')[0] for f in files]

    df = pd.read_parquet(parquet_path, engine='pyarrow')
    # import ipdb; ipdb.set_trace()
    df = preprocess_dataframe(df,indices,'wheat_winter')
    group_index = leave_location_and_time_out_expanding_window(
        data=df, 
        year_col='year',
        space_col='location',
        min_train_years=10,
        test_years=1,
        test_frac=0.3,
        random_state=42
    )

    target_col = "wheat_winter"

    # Select objective function based on model name
    model_name = cfg.model  # or args.model if using argparse

    if model_name == "xgboost":
        objective = make_objective_xgb(df, indices, target_col, group_index)
    elif model_name == "random_forest":
        objective = make_objective_rf(df, indices, target_col, group_index)
    elif model_name == "dnn":
        objective = make_objective_dnn(df, indices, target_col, group_index)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    logger = get_logger("main")

    sampler = TPESampler(seed=42)
    study = optuna.create_study(study_name="xgboost", sampler=sampler, direction="minimize")

    logger.info("Start optimization.")

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    import yaml
    best_params = study.best_trial.params
    with open(f"best_{model_name}_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)


if __name__ == "__main__":
    from dask.distributed import Client, performance_report
    client = Client(n_workers=20, threads_per_worker=2, memory_limit="80GB")  # ~80 logical cores

    import multiprocessing
    multiprocessing.freeze_support()
    main()
