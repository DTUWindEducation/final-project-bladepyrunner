import numpy as np
from src import power_law_calculation


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
