from scipy.stats import weibull_min
from src import compute_speed_direction
import numpy as np


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
