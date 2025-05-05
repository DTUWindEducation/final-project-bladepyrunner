import numpy as np
import xarray as xr
import pandas as pd
import pytest
from src import (
    compute_speed_direction,
    interpolate_wind_data,
    power_law_calculation,
    fit_weibull,
    plot_weibull,
    plot_windrose,
    compute_aep,
    compute_capacity_factor,
    compute_mean_wind_speed,
)

# ---------- Fixtures ----------
@pytest.fixture
def power_curve_file(tmp_path):
    """Creates a temporary CSV file for the turbine power curve."""
    df = pd.DataFrame({
        "wind_speed": [0, 3, 5, 10, 15, 20, 25],
        "power":      [0, 0, 100, 1000, 3000, 5000, 5000]
    })
    file_path = tmp_path / "power_curve.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def dummy_dataset():
    time = pd.date_range("2015-01-01", periods=4, freq="H")
    u10 = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": time})
    v10 = xr.DataArray([0.0, 1.0, 2.0, 3.0], dims="time", coords={"time": time})
    u100 = xr.DataArray([2.0, 3.0, 4.0, 5.0], dims="time", coords={"time": time})
    v100 = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": time})
    ds = xr.Dataset({
        "u10": u10,
        "v10": v10,
        "u100": u100,
        "v100": v100
    }, coords={"latitude": [55.0], "longitude": [10.0]})
    return ds

# ---------- Test compute_speed_direction ----------
def test_compute_speed_direction():
    u = np.array([3.0, 0.0])
    v = np.array([4.0, -1.0])
    speed, direction = compute_speed_direction(u, v)
    assert np.allclose(speed, [5.0, 1.0])
    assert np.all((0 <= direction) & (direction <= 360))

# ---------- Test interpolate_wind_data ----------
def test_interpolate_wind_data(dummy_dataset):
    result = interpolate_wind_data(dummy_dataset, 55.0, 10.0)
    assert 'wind_speed_10' in result
    assert result['wind_speed_10'].shape[0] == 4

# ---------- Test power_law_calculation ----------
def test_power_law_calculation():
    u_ref = np.array([5.0, 6.0])
    u_at_height = power_law_calculation(z_ref=10, u_ref=u_ref, z=100)
    assert np.all(u_at_height > u_ref)

# ---------- Test fit_weibull ----------
def test_fit_weibull(dummy_dataset):
    scale, shape, loc = fit_weibull(dummy_dataset, 55.0, 10.0, 10)
    assert scale > 0
    assert shape > 0
    assert loc == 0

def test_plot_weibull_runs(dummy_dataset):
    wind_speed = np.array([3.5, 4.0, 5.0, 6.0])
    from scipy.stats import weibull_min
    k, A = 2.0, 6.0
    lat, lon = 52.0, 13.0  # Example coordinates (you can use your preferred values)
    height = 10.0          # Set the desired height

    # Should not raise any error
    try:
        plot_weibull(dummy_dataset, lat, lon, height)
    except Exception as e:
        pytest.fail(f"plot_weibull raised an exception: {e}")

# Test for plot_windrose
def test_plot_windrose_runs(dummy_dataset):
    lat, lon = 52.0, 13.0  # Example coordinates
    height = 100.0         # Example height (m)

    # Should not raise any error
    try:
        plot_windrose(dummy_dataset, lat, lon, height)
    except Exception as e:
        pytest.fail(f"plot_windrose raised an exception: {e}")


# Dummy data for testing
def test_compute_aep_valid(dummy_dataset, power_curve_file):
    lat, lon = 52.0, 13.0
    year = 2015
    turbine = 'NREL5MW'

    # Should return a valid AEP value
    try:
        aep = compute_aep(dummy_dataset, lat, lon, power_curve_file, year, turbine)
        assert isinstance(aep, float), f"Expected float, got {type(aep)}"
    except Exception as e:
        pytest.fail(f"compute_aep raised an exception: {e}")

def test_compute_aep_invalid_turbine(dummy_dataset, power_curve_file):
    lat, lon = 52.0, 13.0
    year = 2020
    turbine = 'InvalidTurbine'

    # Should raise ValueError for unsupported turbine
    with pytest.raises(ValueError, match="Unsupported turbine: InvalidTurbine"):
        compute_aep(dummy_dataset, lat, lon, power_curve_file, year, turbine)
    
def test_compute_aep_no_data_for_year(dummy_dataset, power_curve_file):
    lat, lon = 52.0, 13.0
    year = 2025  # Year with no data
    turbine = 'NREL5MW'

    # Should raise ValueError for no data
    with pytest.raises(ValueError, match=f"No data for year {year}"):
        compute_aep(dummy_dataset, lat, lon, power_curve_file, year, turbine)

# Test for compute_capacity_factor
def test_compute_capacity_factor():
    aep = 15.0  # Example AEP in GWh
    rated_power = 5.0  # Example rated power in MW

    # Calculate capacity factor
    cf = compute_capacity_factor(aep, rated_power)

    # Expected capacity factor (AEP / max possible output)
    expected_cf = aep / (rated_power * 8760 / 1000)  # Convert from GWh to MW

    assert np.isclose(cf, expected_cf), f"Expected {expected_cf}, got {cf}"

# Test for compute_mean_wind_speed
def test_compute_mean_wind_speed(dummy_dataset):
    lat, lon = 52.0, 13.0
    height = 100.0  # Example height

    # Should return a valid mean wind speed (float)
    try:
        mean_wind_speed = compute_mean_wind_speed(dummy_dataset, lat, lon, height)
        assert isinstance(mean_wind_speed, float), f"Expected float, got {type(mean_wind_speed)}"
    except Exception as e:
        pytest.fail(f"compute_mean_wind_speed raised an exception: {e}")

def test_compute_mean_wind_speed_invalid_height(dummy_dataset):
    lat, lon = 52.0, 13.0
    height = 50.0  # Invalid height (not 10 or 100)

    # Should return a valid mean wind speed for interpolated height
    try:
        mean_wind_speed = compute_mean_wind_speed(dummy_dataset, lat, lon, height)
        assert isinstance(mean_wind_speed, float), f"Expected float, got {type(mean_wind_speed)}"
    except Exception as e:
        pytest.fail(f"compute_mean_wind_speed raised an exception: {e}")


