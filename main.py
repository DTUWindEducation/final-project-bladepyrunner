import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import datetime
from cftime import num2date
from src import load_nc_files, compute_speed_direction, interpolate_with_xarray, power_law_calculation

#Step 1: Load the wind data from the NetCDF files
print('Loading the wind data...')

data_dir = Path('inputs/WindData')
ds = load_nc_files(data_dir)

u10 = ds['u10'].values
v10 = ds['v10'].values
u100 = ds['u100'].values
v100 = ds['v100'].values
#Step 2: Calculate the wind speed and direction
print('Calculating wind speed and direction...')
wind_speed_10, wind_dir_10 = compute_speed_direction(u10, v10)
wind_speed_100, wind_dir_100 = compute_speed_direction(u100, v100)
#Step 3: Finding Wind Speeds inside the target area using interpolation
print('Interpolating wind speeds to the target area...')
interpolated_data = interpolate_with_xarray(ds, target_lat=7.8, target_lon=55.55)

interpolated_u10 = interpolated_data['u10'].values
interpolated_v10 = interpolated_data['v10'].values
interpolated_u100 = interpolated_data['u100'].values
interpolated_v100 = interpolated_data['v100'].values

#calculating the wind speed and direction for the interpolated data
interpolated_wind_speed_10, interpolated_wind_dir_10 = compute_speed_direction(interpolated_u10, interpolated_v10)
interpolated_wind_speed_100, interpolated_wind_dir_100 = compute_speed_direction(interpolated_u100, interpolated_v100)

#Step 4. Calculating wind speeds at different heights using the power law profile
print('Calculating wind speeds at different heights using the power law profile...')
z_ref = 10  # Reference height (m)
u_ref = wind_speed_10  # Wind speed at reference height (m/s)
z = 80  # Height at which to calculate wind speed (m)
u_80 = power_law_calculation(z_ref, u_ref, z)
