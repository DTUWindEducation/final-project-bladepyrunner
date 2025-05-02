import numpy as np
import xarray as xr
import os


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
    WindData = xr.concat(datasets, dim='time')
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
