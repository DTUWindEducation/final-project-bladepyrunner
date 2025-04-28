import numpy as np
import xarray as xr
import os

def load_nc_files(data_dir):
    datasets = []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".nc"):
            datasets.append(xr.open_dataset(os.path.join(data_dir, file)))
    WindData = xr.concat(datasets, dim='time')
    return WindData



def compute_speed_direction(u, v):
    speed = np.sqrt(u**2 + v**2)
    direction = (np.arctan2(-u, -v) * 180 / np.pi) % 360
    return speed, direction