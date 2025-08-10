import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

def extract_lat_lon_coords(ds):
    """Extract latitude and longitude coordinate names from an xarray dataset."""
    lat = None
    lon = None
    for possible_lat in ['lat', 'latitude']:
        if possible_lat in ds.coords:
            lat = possible_lat
            break
    for possible_lon in ['lon', 'longitude']:
        if possible_lon in ds.coords:
            lon = possible_lon
            break
    if lat is None or lon is None:
        raise ValueError("Latitude and/or longitude coordinates not found in dataset.")
    return lat, lon

def check_dimensions(da):
    """Check if data array has more than two spatial dimensions (excluding time)."""
    non_spatial_dims = {'time'}
    spatial_dims = [dim for dim in da.dims if dim not in non_spatial_dims]
    if len(spatial_dims) != 2:
        raise ValueError(f"Data has unsupported number of spatial dimensions: {spatial_dims}")
    return spatial_dims

def add_shapefile(ax, shapefile_path=None):
    """Add shapefile or fallback to coastlines."""
    if shapefile_path:
        try:
            gdf = gpd.read_file(shapefile_path)
            gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, transform=ccrs.PlateCarree())
        except Exception as e:
            print(f"Failed to load shapefile: {e}. Falling back to coastlines.")
            ax.coastlines()
    else:
        ax.coastlines()

def plot_xarray_data(ds, var_name=None, shapefile_path=None, cmap='viridis'):
    """
    Plot 2D data from xarray Dataset using pcolormesh.
    """
    if var_name is None:
        var_name = list(ds.data_vars)[0]

    da = ds[var_name]

    # Validate dimensions
    check_dimensions(da)

    # Extract coordinates
    lat_name, lon_name = extract_lat_lon_coords(ds)
    lat = ds[lat_name]
    lon = ds[lon_name]

    # Setup plot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
    mesh = ax.pcolormesh(lon, lat, da.squeeze(), cmap=cmap, transform=ccrs.PlateCarree())
    plt.colorbar(mesh, ax=ax, orientation='vertical', label=var_name)

    # Add overlay
    add_shapefile(ax, shapefile_path)

    ax.set_title(f"{var_name} plotted with pcolormesh")
    plt.tight_layout()
    plt.show()
