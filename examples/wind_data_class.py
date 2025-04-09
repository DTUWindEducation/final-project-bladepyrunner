
import xarray as xr
import os

class WindData:
    """
    A class to load ERA5 wind data (NetCDF) for a given file or directory.

    """

    def __init__(self, path):
        """
        Initialize the WindData object and load the data from the specified path.

        """
        self.data = self.load_data(path)

    def load_data(self, path):
        """
        Loads wind data from a NetCDF file or all NetCDF files in a folder.

        """
        ds = xr.open_dataset(path, engine='netcdf4')

        return ds




#loading data
path_to_data = r"inputs\WindData\1997-1999.nc" 

wind = WindData(path_to_data)

print(wind.data)
