import pandas as pd
import xarray as xr
from pathlib import Path
import datetime
from cftime import num2date

# Define the folder path containing the NetCDF files
folder_path = Path('inputs\WindData')
datasets = []  # List to store datasets

for file in folder_path.iterdir():
    ds = xr.open_dataset(file)  # Open each NetCDF file
    datasets.append(ds)  # Store dataset in list

# Concatenate along a specific dimension (e.g., 'time' if it exists)
combined_ds = xr.concat(datasets, dim='valid_time')

# Save or process the combined dataset
# combined_ds.to_netcdf("merged_output.nc")  # Save to a new NetCDF file

# Display the combined dataset
print(combined_ds)