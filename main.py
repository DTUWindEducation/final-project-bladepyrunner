import pandas as pd # import libraries
import netCDF4 # import libraries
import pathlib

import datetime
from cftime import num2date

# Read the data
# folder_path = pathlib.Path('inputs\WindData')
# results = [] # create an empty list to store the results

# for file in folder_path.iterdir():
#     nc_data = netCDF4.Dataset(file, mode='r')
    


import xarray as xr
from pathlib import Path

folder_path = Path('inputs\WindData')  # Replace with your folder path
datasets = []  # List to store datasets

for file in folder_path.iterdir():
    ds = xr.open_dataset(file)  # Open each NetCDF file
    datasets.append(ds)  # Store dataset in list

# Concatenate along a specific dimension (e.g., 'time' if it exists)
combined_ds = xr.concat(datasets, dim='time')

# Save or process the combined dataset
# combined_ds.to_netcdf("merged_output.nc")  # Save to a new NetCDF file

# Display the combined dataset
print(combined_ds)