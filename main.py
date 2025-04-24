import pandas as pd
import xarray as xr
from pathlib import Path
import datetime
from cftime import num2date
from src.__init__ import load_nc_files 
from src.__init__ import compute_speed_direction

# Load the NetCDF file
data_dir = Path('inputs/WindData')
ds = load_nc_files(data_dir)

u10 = ds['u10'].values
v10 = ds['v10'].values

wind_speed, wind_dir = compute_speed_direction(u10, v10)

location_1 = wind_speed[wind_speed['location'] == (7.75, 55.5)]