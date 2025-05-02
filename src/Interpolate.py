import xarray as xr
from src import compute_speed_direction


def interpolate_wind_data(ds, target_lat, target_lon):
    """
    Interpolates wind data to a target latitude and longitude.

    Parameters:
    - ds (xarray.Dataset): The dataset containing wind data.
    - target_lat (float): The target latitude for interpolation.
    - target_lon (float): The target longitude for interpolation.

    Outputs:
    - wind_ds (xarray.Dataset): A dataset containing interpolated
      wind speed and direction at 10m and 100m.
    """

    # Interpolate to the target location
    interpolated_ds = ds.interp(latitude=target_lat, longitude=target_lon)

    # Extract wind components (these retain xarray structure and metadata)
    u10 = interpolated_ds['u10']
    v10 = interpolated_ds['v10']
    u100 = interpolated_ds['u100']
    v100 = interpolated_ds['v100']


    # Compute wind speed and direction
    wind_speed_10, wind_dir_10 = compute_speed_direction(u10.values, v10.values)
    wind_speed_100, wind_dir_100 = compute_speed_direction(u100.values, v100.values)

    # Wrap them in DataArrays, keeping the same dimensions and coordinates as u10
    wind_ds = xr.Dataset(
        data_vars={
            'wind_speed_10': xr.DataArray(wind_speed_10, dims=u10.dims, coords=u10.coords),
            'wind_dir_10': xr.DataArray(wind_dir_10, dims=u10.dims, coords=u10.coords),
            'wind_speed_100': xr.DataArray(wind_speed_100, dims=u100.dims, coords=u100.coords),
            'wind_dir_100': xr.DataArray(wind_dir_100, dims=u100.dims, coords=u100.coords)
        }
    )

    return wind_ds
