import pytest
import numpy as np
import xarray as xr
from src import (
    load_nc_files, 
    compute_speed_direction, 
    interpolate_wind_data, 
    power_law_calculation, 
    fit_weibull, 
    plot_weibull, 
    plot_windrose, 
    compute_aep, 
    compute_capacity_factor, 
    compute_mean_wind_speed
)
@pytest.fixture
def mock_data():
    # Create a simple mock xarray dataset for testing
    latitudes = np.linspace(10, 20, 5)
    longitudes = np.linspace(30, 40, 5)
    times = pd.date_range('2020-01-01', periods=10, freq='D')
    
    u10 = np.random.rand(10, 5, 5)
    v10 = np.random.rand(10, 5, 5)
    u100 = np.random.rand(10, 5, 5)
    v100 = np.random.rand(10, 5, 5)
    
    ds = xr.Dataset(
        {
            'u10': (['time', 'latitude', 'longitude'], u10),
            'v10': (['time', 'latitude', 'longitude'], v10),
            'u100': (['time', 'latitude', 'longitude'], u100),
            'v100': (['time', 'latitude', 'longitude'], v100),
        },
        coords={
            'time': times,
            'latitude': latitudes,
            'longitude': longitudes,
        }
    )
    return ds

def test_compute_speed_direction():
    u = np.array([3, 4])
    v = np.array([4, 3])
    
    speed, direction = compute_speed_direction(u, v)
    
    assert np.allclose(speed, [5, 5])
    assert np.allclose(direction, [45, 36.86989765])

def test_interpolate_wind_data(mock_data):
    lat, lon = 15, 35
    wind_ds = interpolate_wind_data(mock_data, lat, lon)
    
    assert 'wind_speed_10' in wind_ds
    assert 'wind_dir_10' in wind_ds
    assert 'wind_speed_100' in wind_ds
    assert 'wind_dir_100' in wind_ds

def test_power_law_calculation():
    u_ref = 10
    z_ref = 10
    z = 50
    alpha = 1 / 7
    
    speed = power_law_calculation(z_ref, u_ref, z, alpha)
    
    assert speed > u_ref  # Wind speed at height z should be higher

def test_fit_weibull(mock_data):
    lat, lon = 15, 35
    height = 10  # 10 meters
    
    scale, shape, loc = fit_weibull(mock_data, lat, lon, height)
    
    assert isinstance(scale, float)
    assert isinstance(shape, float)
    assert isinstance(loc, float)

def test_plot_weibull(mock_data):
    lat, lon = 15, 35
    height = 10  # 10 meters
    
    # Test that the plot function runs without errors
    try:
        plot_weibull(mock_data, lat, lon, height)
    except Exception as e:
        pytest.fail(f"Plotting failed: {e}")

def test_plot_windrose(mock_data):
    lat, lon = 15, 35
    height = 10  # 10 meters
    
    # Test that the plot function runs without errors
    try:
        plot_windrose(mock_data, lat, lon, height)
    except Exception as e:
        pytest.fail(f"Plotting failed: {e}")

def test_compute_aep(mock_data):
    lat, lon = 15, 35
    power_curve_file = "path/to/power_curve.csv"
    year = 2020
    turbine = "NREL5MW"
    
    # Test AEP computation with a mock power curve
    try:
        aep = compute_aep(mock_data, lat, lon, power_curve_file, year, turbine)
    except Exception as e:
        pytest.fail(f"AEP calculation failed: {e}")
    
    assert isinstance(aep, float)
    assert aep >= 0

def test_compute_capacity_factor():
    aep = 500  # GWh
    rated_power = 5  # MW
    
    cf = compute_capacity_factor(aep, rated_power)
    
    assert 0 <= cf <= 1  # Capacity factor must be between 0 and 1

def test_compute_mean_wind_speed(mock_data):
    lat, lon = 15, 35
    height = 10
    
    mean_wind_speed = compute_mean_wind_speed(mock_data, lat, lon, height)
    
    assert isinstance(mean_wind_speed, float)
    assert mean_wind_speed > 0