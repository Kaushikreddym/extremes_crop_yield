#!/usr/bin/env python
# coding: utf-8

# In[2]:


from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import pandas as pd

def ext_loess(df, frac=0.4):
    """
    Adds LOESS smoothed mean yield and yield anomaly columns to the DataFrame.

    Parameters:
    - df (pd.DataFrame): must contain 'lat', 'lon', 'date', 'yield'
    - frac (float): smoothing parameter for LOESS

    Returns:
    - df with added columns: 'mean_yield', 'yield_anomaly'
    """
    df = df.copy()

    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['time']).dt.year

    df['mean_yield'] = np.nan
    df['yield_anomaly'] = np.nan
    # df['yield'] = np.nan

    for (lat, lon), group in df.groupby(['lat', 'lon']):
        sorted_group = group.sort_values('year')
        years = sorted_group['year'].values
        yields = sorted_group['yield'].values

        # Apply LOESS even if < 5 years
        loess_result = lowess(endog=yields, exog=years, frac=frac, return_sorted=False)

        df.loc[sorted_group.index, 'mean_yield'] = loess_result
        df.loc[sorted_group.index, 'yield_anomaly'] = yields - loess_result
        # df.loc[sorted_group.index, 'yield'] = yields

    return df


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
from glob import glob
import argparse
from tqdm import tqdm
from sklearn.neighbors import BallTree
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import optuna
from optuna.samplers import TPESampler
import logging
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.abspath('..'))

from datasets.datasets import *
from datasets.indices import *
from utils.utils import *
from utils.ml import *

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = 'Times New Roman'

region='europe'
ind='*'
files = glob(f'data/10km/{ind}_mswx_{region}_1989-01-01_2024-12-31_YS_10km.zarr/')
indices = [f.split('/')[-2].split('_mswx_')[0] for f in files]
df = pd.read_parquet(f'data/combined/index_yield_mswx_{region}_1989-01-01_2024-12-31_YS_10km.parquet', engine='pyarrow')

df = df.rename(columns={'wheat_winter': 'yield'})
df = preprocess_dataframe(df,indices,'yield')

df = ext_loess(df, frac=0.4)

data = df

n_train = int(data['year'].nunique() * 0.7)
n_test = data['year'].nunique() - n_train

train_years = sorted(data['year'].unique())[:n_train]
test_years = sorted(data['year'].unique())[-n_test:]

train_data = data[data['year'].isin(train_years)]
test_data = data[data['year'].isin(test_years)]


print(train_data.shape, test_data.shape)

with open("best_xgboost_params.yaml", "r") as f:
    best_params = yaml.safe_load(f)

# Train the final model
X_train = train_data.loc[:,indices]
X_train_info = train_data[['lat','lon','time','location','year']]
y_train_anomaly = train_data.loc[:,['yield_anomaly']]
y_train = train_data.loc[:,['yield']]

X_test = test_data.loc[:,indices]
X_test_info = test_data[['lat','lon','time','location','year']]
y_test_anomaly = test_data.loc[:,['yield_anomaly']]
y_test = test_data.loc[:,['yield']]

# Standardize the data
X_train_scaled, X_test_scaled = robust_scale_train_test(X_train, X_test)


# Update the preds_df
test_preds = X_test_info.copy()
train_preds = X_train_info.copy()

test_preds['observed_yield'] = y_test
train_preds['observed_yield'] = y_train

test_preds['observed_yield_anomaly'] = y_test_anomaly
train_preds['observed_yield_anomaly'] = y_train_anomaly

# Create a model instance
model = XGBRegressor(**best_params, tree_method="hist", random_state=42,predictor="gpu_predictor", device="cuda")

# Fit the model
model.fit(X_train_scaled, y_train.values)

# Predict on the test set and train set
y_pred_test = model.predict(X_test_scaled)
y_pred_train = model.predict(X_train_scaled)

test_preds['predicted_yield'] = y_pred_test
train_preds['predicted_yield'] = y_pred_train

# Fit the model
model.fit(X_train_scaled, y_train_anomaly.values)

# Predict on the test set and train set
y_pred_test = model.predict(X_test_scaled)
y_pred_train = model.predict(X_train_scaled)

test_preds['predicted_yield_anomaly'] = y_pred_test
train_preds['predicted_yield_anomaly'] = y_pred_train

print(train_preds.shape, test_preds.shape)
train_preds.head()


# In[4]:


import xarray as xr
import xesmf as xe
import numpy as np

import hydra
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import colormaps as cmaps
from utils.plotting import check_dimensions, extract_lat_lon_coords, add_shapefile
from datasets.datasets import *
# Initialize Hydra without running main
hydra.core.global_hydra.GlobalHydra.instance().clear()  # clear previous hydra state if any
hydra.initialize(config_path="conf")  # relative to your notebook working dir

cfg = hydra.compose(config_name="config")

region = 'europe'
cfg.region = region
ind = 'max_cdd'
yield_data = load_GDHY(cfg.mappings.GDHY.path, cfg.bounds[cfg.region])


# In[5]:


import xarray as xr
import matplotlib.pyplot as plt

# Assume `da` is your xarray.DataArray with dims: time, lat, lon

# Count non-NaN values along the time dimension
count = yield_data['wheat_winter'].count(dim='time')  # result is 2D: (lat, lon)
# Create a masked array: mask zeros
masked = np.ma.masked_where(count == 0, count)

# Create a colormap with white for masked (zero) values
cmap = plt.cm.viridis
cmap.set_bad(color='white')  # white for masked values

# Plot
plt.figure(figsize=(10, 6))
masked_plot = plt.pcolormesh(count['lon'], count['lat'], masked, cmap=cmap)
plt.colorbar(masked_plot, label='Non-NaN Count')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Non-NaN Value Count (White = 0)')
plt.show()


# In[6]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

def plot_country_yield_timeseries(df, shapefile_path, country_name):
    """
    Plot observed and predicted yield time series for a given country.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'lat', 'lon', 'year', 'yield', 'predicted_yield'
    - shapefile_path (str): Path to country shapefile
    - country_name (str): Name of the country to plot (should match 'ADMIN' field in shapefile)
    """
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)

    # Create GeoDataFrame from df
    df = df.copy()
    df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    df_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=gdf.crs)

    # Spatial join to get country
    df_with_country = gpd.sjoin(
        df_gdf, gdf[['geometry', 'ADMIN']], how='inner', predicate='within'
    )

    # Aggregate by country and year
    agg_df = df_with_country.groupby(['ADMIN', 'year'])[['observed_yield', 'predicted_yield']].mean().reset_index()

    # Subset for selected country
    subset = agg_df[agg_df['ADMIN'] == country_name]
    if subset.empty:
        print(f"No data found for country: {country_name}")
        return
    
    plt.figure(figsize=(10, 5))

    # Vertical line at 2008
    plt.axvline(x=2008, color='gray', linestyle='--')
    plt.text(2006.5, subset[['observed_yield', 'predicted_yield']].max().max()*0.9, 'Train', 
             horizontalalignment='right', fontsize=14, color='gray')
    plt.text(2009.5, subset[['observed_yield', 'predicted_yield']].max().max()*0.9, 'Test', 
             horizontalalignment='left', fontsize=14, color='gray')
    
    plt.plot(subset['year'], subset['observed_yield'], label='Observed Yield',marker='o')
    plt.plot(subset['year'], subset['predicted_yield'], label='Predicted Yield',marker='o')
    plt.title(f'Yield Time Series - {country_name}')
    plt.xlabel('Year')
    plt.ylabel('Yield')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_country_yield_timeseries(
    df=pd.concat([train_preds,test_preds]),
    shapefile_path='/beegfs/muduchuru/data/shp/ne/ne_10m_admin_0_countries.shp',
    country_name='Germany'
)

plot_country_yield_timeseries(
    df=pd.concat([train_preds,test_preds]),
    shapefile_path='/beegfs/muduchuru/data/shp/ne/ne_10m_admin_0_countries.shp',
    country_name='Italy'
)



# In[7]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

def plot_country_yield_timeseries(df, shapefile_path, country_name):
    """
    Plot observed and predicted yield time series for a given country.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'lat', 'lon', 'year', 'yield', 'predicted_yield'
    - shapefile_path (str): Path to country shapefile
    - country_name (str): Name of the country to plot (should match 'ADMIN' field in shapefile)
    """
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)

    # Create GeoDataFrame from df
    df = df.copy()
    df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    df_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=gdf.crs)

    # Spatial join to get country
    df_with_country = gpd.sjoin(
        df_gdf, gdf[['geometry', 'ADMIN']], how='inner', predicate='within'
    )

    # Aggregate by country and year
    agg_df = df_with_country.groupby(['ADMIN', 'year'])[['observed_yield_anomaly', 'predicted_yield_anomaly']].mean().reset_index()

    # Subset for selected country
    subset = agg_df[agg_df['ADMIN'] == country_name]
    if subset.empty:
        print(f"No data found for country: {country_name}")
        return
    
    plt.figure(figsize=(10, 5))

    # Vertical line at 2008
    plt.axvline(x=2008, color='gray', linestyle='--')
    plt.text(2006.5, subset[['observed_yield_anomaly', 'predicted_yield_anomaly']].max().max()*0.9, 'Train', 
             horizontalalignment='right', fontsize=14, color='gray')
    plt.text(2009.5, subset[['observed_yield_anomaly', 'predicted_yield_anomaly']].max().max()*0.9, 'Test', 
             horizontalalignment='left', fontsize=14, color='gray')
    
    plt.plot(subset['year'], subset['observed_yield_anomaly'], label='Observed Yield',marker='o')
    plt.plot(subset['year'], subset['predicted_yield_anomaly'], label='Predicted Yield',marker='o')
    plt.title(f'Yield Time Series - {country_name}')
    plt.xlabel('Year')
    plt.ylabel('Yield')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_country_yield_timeseries(
    df=pd.concat([train_preds,test_preds]),
    shapefile_path='/beegfs/muduchuru/data/shp/ne/ne_10m_admin_0_countries.shp',
    country_name='Germany'
)

plot_country_yield_timeseries(
    df=pd.concat([train_preds,test_preds]),
    shapefile_path='/beegfs/muduchuru/data/shp/ne/ne_10m_admin_0_countries.shp',
    country_name='Italy'
)



# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Assume df has columns: daily_pr_intensity, maximum_cdd, yield
x = df['daily_pr_intensity']
y = df['tg_days_above_10degC']
z = df['yield']

# Set up a grid with 2D histogram center, and KDEs on top/right
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                       wspace=0.05, hspace=0.05)

ax_main = plt.subplot(gs[1, 0])  # main heatmap
ax_xkde = plt.subplot(gs[0, 0], sharex=ax_main)  # x-axis KDE
ax_ykde = plt.subplot(gs[1, 1], sharey=ax_main)  # y-axis KDE

# Compute 2D histogram bins
hb = ax_main.hexbin(x, y, C=z, gridsize=40, reduce_C_function=np.mean, cmap='viridis')
cb = fig.colorbar(hb, ax=ax_main, label='Mean Yield')
ax_main.set_xlabel('Daily Precipitation Intensity')
ax_main.set_ylabel('tg_days_above_10degC')

# KDE plots on top and right
sns.kdeplot(x=x, ax=ax_xkde, fill=True, linewidth=1.5)
ax_xkde.axis('off')  # hide axes ticks and labels

sns.kdeplot(y=y, ax=ax_ykde, fill=True, linewidth=1.5)
ax_ykde.axis('off')  # hide axes ticks and labels

plt.show()


# In[46]:


df.columns


# In[9]:


from sklearn.metrics import r2_score#mean_absolute_percentage_error as r2_score

r2_global = r2_score(test_preds["observed_yield"], test_preds["predicted_yield"])
print("Global R²:", r2_global)

r2_by_time = test_preds.groupby("time").apply(
    lambda g: r2_score(g["observed_yield"], g["predicted_yield"])
)
r2_by_time.name = "r2"
print(r2_by_time)

r2_by_location = test_preds.groupby(["lat", "lon"]).apply(
    lambda g: r2_score(g["observed_yield"], g["predicted_yield"])
)
r2_by_location.name = "r2"
print(r2_by_location)


# In[10]:


from sklearn.metrics import r2_score#mean_absolute_percentage_error as r2_score

r2_global = r2_score(test_preds["observed_yield_anomaly"], test_preds["predicted_yield_anomaly"])
print("Global R²:", r2_global)

r2_by_time = test_preds.groupby("time").apply(
    lambda g: r2_score(g["observed_yield_anomaly"], g["predicted_yield_anomaly"])
)
r2_by_time.name = "r2"
print(r2_by_time)

r2_by_location = test_preds.groupby(["lat", "lon"]).apply(
    lambda g: r2_score(g["observed_yield_anomaly"], g["predicted_yield_anomaly"])
)
r2_by_location.name = "r2"
print(r2_by_location)


# In[13]:


from sklearn.metrics import r2_score#mean_absolute_percentage_error as r2_score

preds = test_preds
r2_global = r2_score(preds["observed_yield_anomaly"], preds["predicted_yield_anomaly"])
print("Global R²:", r2_global)

r2_by_time = preds.groupby("time").apply(
    lambda g: r2_score(g["observed_yield_anomaly"], g["predicted_yield_anomaly"])
)
r2_by_time.name = "r2"
print(r2_by_time)

r2_by_location = preds.groupby(["lat", "lon"]).apply(
    lambda g: r2_score(g["observed_yield_anomaly"], g["predicted_yield_anomaly"])
)
r2_by_location.name = "r2"
print(r2_by_location)

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import colormaps as cmaps
import xarray as xr

from utils.plotting import check_dimensions, extract_lat_lon_coords, add_shapefile

# Convert DataFrame to xarray DataArray
r2_map = r2_by_location.reset_index().pivot(index="lat", columns="lon", values="r2")

r2_xr = xr.DataArray(
    r2_map.values,
    coords={"lat": r2_map.index.values, "lon": r2_map.columns.values},
    dims=["lat", "lon"],
    name="r2_time"
)

ptot_dsets = [r2_xr]
dset_names = ['r2_time']
cmap = plt.cm.RdYlGn_r
shapefile_path = None

projection = ccrs.LambertConformal(central_longitude=10, central_latitude=50)

# Create subplot
fig, axes = plt.subplots(
    nrows=1, ncols=1,
    subplot_kw={'projection': projection},
    figsize=(10, 10)
)

# Handle axes being a single object
if isinstance(axes, plt.Axes):
    axes = [axes]

# Plot
for i, da in enumerate(ptot_dsets):
    check_dimensions(da)
    lat_name, lon_name = extract_lat_lon_coords(da)
    lat = da[lat_name]
    lon = da[lon_name]

    ax = axes[i]
    mesh = ax.pcolormesh(
        lon, lat, da.squeeze(),
        cmap=cmap, vmin=0, vmax=0.4,
        transform=ccrs.PlateCarree()
    )
    ax.coastlines(resolution="10m")
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', label=dset_names[i], shrink=0.5)
    add_shapefile(ax, shapefile_path)
    ax.set_title(f"{dset_names[i]}")

plt.tight_layout()
plt.show()


# In[35]:


preds = train_preds
for year in preds.year.unique():
    map = preds[
        preds['year']==year
        ].reset_index().pivot_table(
            index="lat", columns="lon", values="predicted_yield_anomaly"
            )
    pred_xr = xr.DataArray(
        map.values,
        coords={"lat": map.index.values, "lon": map.columns.values},
        dims=["lat", "lon"],
        name="Predicted Yield"
    )
    map = preds[
        preds['year']==year
        ].reset_index().pivot_table(
            index="lat", columns="lon", values="observed_yield_anomaly"
            )
    obs_xr = xr.DataArray(
        map.values,
        coords={"lat": map.index.values, "lon": map.columns.values},
        dims=["lat", "lon"],
        name="Observed Yield"
    )

    ptot_dsets = [obs_xr, pred_xr]
    dset_names = ['Obs. Yield', 'Pred. Yield']
    cmap = plt.cm.RdYlGn_r
    shapefile_path = None

    projection = ccrs.LambertConformal(central_longitude=10, central_latitude=50)

    # Create subplot
    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        subplot_kw={'projection': projection},
        figsize=(10, 10)
    )

    # Handle axes being a single object
    if isinstance(axes, plt.Axes):
        axes = [axes]

    # Plot
    for i, da in enumerate(ptot_dsets):
        check_dimensions(da)
        lat_name, lon_name = extract_lat_lon_coords(da)
        lat = da[lat_name]
        lon = da[lon_name]

        ax = axes[i]
        mesh = ax.pcolormesh(
            lon, lat, da.squeeze(),
            cmap=cmap, vmin=-1, vmax=1,
            transform=ccrs.PlateCarree()
        )
        ax.coastlines(resolution="10m")
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', label=dset_names[i], shrink=0.5)
        add_shapefile(ax, shapefile_path)
        ax.set_title(f"{dset_names[i]} | {year}")

    plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:




