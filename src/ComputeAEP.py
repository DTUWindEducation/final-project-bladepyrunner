import pandas as pd
import numpy as np
from src import power_law_calculation, interpolate_wind_data
from scipy.interpolate import interp1d


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
