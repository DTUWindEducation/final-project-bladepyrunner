import xarray as xr

def interpolate_with_xarray(ds, target_lat, target_lon):
    """
    Interpolate entire dataset to a target location using xarray's built-in interpolation.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing wind data with coordinates 'latitude' and 'longitude'
    target_lat, target_lon : float
        Coordinates of the target point
        
    Returns:
    --------
    xarray.Dataset
        Dataset interpolated to the target location
    """
    # Create a new dataset with a single point at the target location
    return ds.interp(latitude=target_lat, longitude=target_lon)