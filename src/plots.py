import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import weibull_min
from windrose import WindroseAxes
from src import compute_speed_direction, power_law_calculation


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
    Interpolates wind data to a location and height, computes wind speed and direction, and plots a windrose.
    
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
    