import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from windrose import WindroseAxes
from scipy.interpolate import interp1d
def load_nc_files(data_dir):
    """
    Load NetCDF files from a directory and concatenate them into
    a single xarray dataset.
    Args:
        data_dir (str): Path to the directory containing NetCDF files.
        Returns:
        xarray.Dataset: Concatenated dataset containing all the NetCDF files.
    """
    datasets = []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".nc"):
            datasets.append(xr.open_dataset(os.path.join(data_dir, file)))
    WindData = xr.concat(datasets, dim='time', coords="minimal", compat='override')
    return WindData


def compute_speed_direction(u, v):
    """
    Computes wind speed and direction from u and v components.

    Parameters:
    - u (numpy.ndarray): U component of wind (m/s)
    - v (numpy.ndarray): V component of wind (m/s)

    Outputs:
    - speed (numpy.ndarray): Wind speed (m/s)
    - direction (numpy.ndarray): Wind direction (degrees)
    """
    # Calculate wind speed and direction
    speed = np.sqrt(u**2 + v**2)
    direction = (np.arctan2(-u, -v) * 180 / np.pi) % 360
    return speed, direction


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


def power_law_calculation(z_ref, u_ref, z, alpha=1/7):
    """
    Calculate wind speed at height z using the power law profile.
    Parameters:
    z_ref : float
        Reference height (m)
    u_ref : float
        Wind speed at reference height (m/s)
    z : float
        Height at which to calculate wind speed (m)
    alpha : float
        Power law exponent (dimensionless)
    Returns : float
        Wind speed at height z (m/s)
    """
    # Check if the inputs are valid
    return u_ref * (z / z_ref) ** alpha


def fit_weibull(ds, lat, lon, height):
    """
    Interpolates wind data to a specified location and height,
    computes wind speed, and fits a Weibull distribution.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing wind components (e.g. 'u10', 'v10', 'u100', 'v100')
    lat : float
        Latitude of the target location
    lon : float
        Longitude of the target location
    height : int
        Desired height (must be 10 or 100)

    Returns:
    --------
    tuple:
        (scale, shape, loc) - Weibull parameters
    """
    # Fit the Weibull distribution to the wind speed data
    u_var = f'u{height}'
    v_var = f'v{height}'

    # interpolate data set
    interpolated_ds = ds.interp(latitude=lat, longitude=lon)

    # extracting wind components
    u = interpolated_ds[u_var].values
    v = interpolated_ds[v_var].values

    # remove nan values
    clean_u = u[~np.isnan(u)]
    clean_v = v[~np.isnan(v)]

    speed, _ = compute_speed_direction(clean_u, clean_v)
    # fitting weibull distribution
    shape, loc, scale = weibull_min.fit(speed, floc=0)
    return scale, shape, loc


def plot_weibull(ds, lat, lon, height, alpha=1/7):
    """
    Interpolates wind data to a location and height,
    fits Weibull distribution, and plots it.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing u10, v10, u100, v100
    lat, lon : float
        Geographic coordinates
    height : float
        Target height for wind speed calculation (in meters)
    alpha : float
        Power law exponent (default: 1/7)
    """
    # Interpolate to location
    interpolated = ds.interp(latitude=lat, longitude=lon)

    # Get wind components
    u10, v10 = interpolated['u10'].values, interpolated['v10'].values
    u100, v100 = interpolated['u100'].values, interpolated['v100'].values

    # Calculate wind components at desired height
    if abs(height - 10) < 1e-2:
        u, v = u10, v10
    elif abs(height - 100) < 1e-2:
        u, v = u100, v100
    else:
        u = power_law_calculation(u10, 10, height, alpha)
        v = power_law_calculation(v10, 10, height, alpha)

    # Compute wind speed
    wind_speed, _ = compute_speed_direction(u, v)

    # Remove NaNs and infs
    wind_speed = wind_speed[np.isfinite(wind_speed)]

    # Fit Weibull
    shape, loc, scale = weibull_min.fit(wind_speed, floc=0)

    # Plot
    x = np.linspace(0, max(wind_speed), 100)
    pdf = weibull_min.pdf(x, shape, scale=scale)
    plt.hist(wind_speed, bins=30, density=True, alpha=0.6, color='g', label='Wind data')
    plt.plot(x, pdf, 'r-', lw=2, label=f'Weibull fit (k={shape:.2f}, A={scale:.2f})')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Probability Density')
    plt.title(f'Weibull Distribution at {height} m (Lat {lat}, Lon {lon})')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_windrose(ds, lat, lon, height, alpha=1/7):
    """
    Interpolates wind data to a location and height, computes wind speed
    and direction, and plots a windrose.

    Parameters:
    ds: xarray.Dataset
        Dataset containing wind data (u10, v10, u100, v100)
        lat: float
            Latitude of the target location
        lon: float
            Longitude of the target location
        height: float
            Height at which to calculate wind speed and direction (in meters)
        alpha: float
            Power law exponent (default: 1/7)

    Outputs:
    -Windrose plot at the specified location and height.
    """
    # Interpolate u and v components at both 10m and 100m
    interpolated = ds.interp(latitude=lat, longitude=lon)
    u10, v10 = interpolated['u10'].values, interpolated['v10'].values
    u100, v100 = interpolated['u100'].values, interpolated['v100'].values

    # Calculate wind speed at 10m and 100m
    speed_10, _ = compute_speed_direction(u10, v10)
    speed_100, _ = compute_speed_direction(u100, v100)

    # Linearly interpolate wind speed to target height using power law
    if abs(height - 10) < 1e-2:
        speed_target = speed_10
        u_target, v_target = u10, v10
    elif abs(height - 100) < 1e-2:
        speed_target = speed_100
        u_target, v_target = u100, v100
    else:
        speed_target = power_law_calculation(speed_10, 10, height, alpha)
        # Also scale u and v separately for direction to remain consistent
        u_target = power_law_calculation(u10, 10, height, alpha)
        v_target = power_law_calculation(v10, 10, height, alpha)

    # Compute wind direction
    _, wind_dir = compute_speed_direction(u_target, v_target)

    # Clean NaNs
    mask = ~np.isnan(speed_target) & ~np.isnan(wind_dir)
    speed_target = speed_target[mask]
    wind_dir = wind_dir[mask]

    # Plot windrose
    ax = WindroseAxes.from_ax()
    ax.bar(wind_dir, speed_target, normed=True, opening=0.8, edgecolor='white')
    ax.set_title(f'Windrose at {lat:.2f}°, {lon:.2f}° @ {height} m', fontsize=12)
    ax.set_legend()
    plt.show()


def compute_aep(ds, lat, lon, power_curve_file, year, turbine):
    """
    Computes the Annual Energy Production (AEP) for a given location and year.

    Parameters:
    - ds (xarray.Dataset): The dataset containing wind data.
    - lat (float): Latitude of the target location.
    - lon (float): Longitude of the target location.
    - power_curve_file (str): Path to the power curve CSV file.
    - year (int): Year for which to compute AEP.
    - turbine (str): Type of turbine ('NREL5MW' or 'NREL15MW').

    Outputs:
    - aep_mwh (float): Annual Energy Production in GWh.
    """
    # Validate turbine
    turbine_heights = {'NREL5MW': 90, 'NREL15MW': 150}
    if turbine not in turbine_heights:
        raise ValueError(f"Unsupported turbine: {turbine}")
    height = turbine_heights[turbine]

    # Interpolate to location
    interpolated = interpolate_wind_data(ds, lat, lon)

    # Filter by year
    time = pd.to_datetime(interpolated.time.values)
    year_mask = time.year == year
    if not year_mask.any():
        raise ValueError(f"No data for year {year}")

    wind_speed_10 = interpolated['wind_speed_10'].values[year_mask]
    if np.all(np.isnan(wind_speed_10)):
        raise ValueError("All interpolated wind speeds are NaN at this location/year.")

    # Power law profile to hub height
    wind_speed_hub = power_law_calculation(10, wind_speed_10, height)

    # Load power curve
    df = pd.read_csv(power_curve_file)
    wind_speeds = df.iloc[:, 0].values
    power_outputs = df.iloc[:, 1].values

    # Ensure the curve is sorted for interpolation
    sorted_indices = np.argsort(wind_speeds)
    wind_speeds = wind_speeds[sorted_indices]
    power_outputs = power_outputs[sorted_indices]

    # Interpolate power output
    interp_func = interp1d(wind_speeds, power_outputs, bounds_error=False, fill_value=0.0)
    power_output = interp_func(wind_speed_hub)

    # Remove NaNs just in case
    power_output = np.nan_to_num(power_output)

    # Compute AEP (assuming hourly resolution)
    aep_mwh = np.sum(power_output) / 1000000  # from kWh to GWh

    return aep_mwh


def compute_capacity_factor(aep, rated_power):
    """
    Computes the capacity factor of a wind turbine.

    Parameters:
    - aep (float): Annual Energy Production in GWh
    - rated_power (float): Turbine's rated power in MW

    Returns:
    - Capacity factor (float between 0 and 1)
    """
    max_possible_output = rated_power * 8760 / 1000  # Convert to GWh
    return aep / max_possible_output


def compute_mean_wind_speed(ds, lat, lon, height):
    """
    Compute the mean wind speed at a specific location and height.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing 'u10', 'v10', 'u100', 'v100'.
    lat, lon : float
        Location coordinates.
    height : float
        Height (in meters) at which to compute wind speed
        (10, 100, or in-between).

    Returns:
    --------
    float
        Mean wind speed in m/s.
    """
    # Interpolate to the given location
    point = ds.interp(latitude=lat, longitude=lon)

    # Compute wind speeds at 10 m and 100 m
    ws_10 = np.sqrt(point['u10']**2 + point['v10']**2)
    ws_100 = np.sqrt(point['u100']**2 + point['v100']**2)

    # Interpolate to desired height using power law
    if height == 10:
        ws = ws_10
    elif height == 100:
        ws = ws_100
    else:
        # Power law interpolation
        ws = power_law_calculation(10, ws_10, height)
        # alpha = np.log(ws_100 / ws_10) / np.log(100 / 10)
        # ws = ws_10 * (height / 10) ** alpha

    # Remove invalid values
    ws = ws.where(np.isfinite(ws), drop=True)

    return float(ws.mean().values)