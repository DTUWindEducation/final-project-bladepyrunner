import pytest
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import weibull_min
from __init__ import (
    compute_speed_direction,
    interpolate_wind_data,
    power_law_calculation,
    fit_weibull,
    compute_capacity_factor,
    compute_mean_wind_speed
)

@pytest.fixture
def dummy_dataset():
    time = pd.date_range("2020-01-01", periods=3, freq="H")
    latitude = [50.0]
    longitude = [10.0]
    u10 = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["time"],
                       coords={"time": time})
    v10 = xr.DataArray(np.array([0.0, -1.0, -2.0]), dims=["time"],
                       coords={"time": time})
    ds = xr.Dataset({
        "u10": (["time"], u10),
        "v10": (["time"], v10),
        "u100": (["time"], u10 + 1),
        "v100": (["time"], v10 + 1)
    }, coords={"time": time, "latitude": latitude, "longitude": longitude})
    return ds.expand_dims(latitude=latitude, longitude=longitude)

def test_compute_speed_direction():
    u = np.array([3.0, 0.0])
    v = np.array([4.0, -1.0])
    speed, direction = compute_speed_direction(u, v)
    np.testing.assert_almost_equal(speed, [5.0, 1.0])
    assert np.all((0 <= direction) & (direction <= 360))

def test_interpolate_wind_data(dummy_dataset):
    result = interpolate_wind_data(dummy_dataset, 50.0, 10.0)
    assert 'wind_speed_10' in result
    assert result['wind_speed_10'].shape == (3,)

def test_power_law_calculation():
    speed = power_law_calculation(10, 5.0, 100)
    assert speed > 5.0  # Wind should increase with height

def test_fit_weibull(dummy_dataset):
    scale, shape, loc = fit_weibull(dummy_dataset, 50.0, 10.0, 10)
    assert scale > 0
    assert shape > 0
    assert loc == 0

def test_compute_capacity_factor():
    aep = 20  # GWh
    rated_power = 5  # MW
    cf = compute_capacity_factor(aep, rated_power)
    assert 0 <= cf <= 1

def test_compute_mean_wind_speed(dummy_dataset):
    mean_speed = compute_mean_wind_speed(dummy_dataset, 50.0, 10.0, 50)
    assert isinstance(mean_speed, float)
    assert mean_speed > 0
