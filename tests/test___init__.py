import unittest
import numpy as np
import xarray as xr
import pandas as pd
from unittest.mock import patch

class TestWindAnalysisModule(unittest.TestCase):

    def test_compute_speed_direction(self):
        u = np.array([0, 3, 0])
        v = np.array([0, 4, -4])
        speed, direction = mod.compute_speed_direction(u, v)
        np.testing.assert_array_almost_equal(speed, [0, 5, 4])
        self.assertTrue((0 <= direction).all() and (direction < 360).all())

    def test_power_law_calculation(self):
        result = mod.power_law_calculation(10, 5, 100)
        expected = 5 * (100 / 10) ** (1/7)
        self.assertAlmostEqual(result, expected)

    def test_interpolate_wind_data(self):
        # Synthetic dataset
        ds = xr.Dataset({
            "u10": (("time",), np.array([1.0, 2.0])),
            "v10": (("time",), np.array([3.0, 4.0])),
            "u100": (("time",), np.array([5.0, 6.0])),
            "v100": (("time",), np.array([7.0, 8.0]))
        }, coords={"time": [0, 1], "latitude": 45.0, "longitude": -60.0})
        
        ds = ds.expand_dims(dim=["latitude", "longitude"])
        result = mod.interpolate_wind_data(ds, 45.0, -60.0)
        self.assertIn("wind_speed_10", result)
        self.assertIn("wind_dir_10", result)

    def test_fit_weibull(self):
        ds = xr.Dataset({
            "u10": (("time",), np.array([3.0, 4.0, 5.0])),
            "v10": (("time",), np.array([4.0, 3.0, 0.0]))
        }, coords={"time": pd.date_range("2020-01-01", periods=3), "latitude": 30.0, "longitude": 40.0})
        ds = ds.expand_dims(dim=["latitude", "longitude"])
        shape, scale, loc = mod.fit_weibull(ds, 30.0, 40.0, 10)
        self.assertTrue(scale > 0 and shape > 0)

    def test_compute_capacity_factor(self):
        result = mod.compute_capacity_factor(aep=100, rated_power=5)
        expected = 100 / (5 * 8760 / 1000)
        self.assertAlmostEqual(result, expected)

    def test_compute_mean_wind_speed(self):
        ds = xr.Dataset({
            "u10": (("time",), np.array([1.0, 2.0, 3.0])),
            "v10": (("time",), np.array([1.0, 2.0, 2.0])),
            "u100": (("time",), np.array([2.0, 3.0, 4.0])),
            "v100": (("time",), np.array([2.0, 3.0, 3.0]))
        }, coords={"time": pd.date_range("2022-01-01", periods=3), "latitude": 45.0, "longitude": 60.0})
        ds = ds.expand_dims(dim=["latitude", "longitude"])
        result = mod.compute_mean_wind_speed(ds, 45.0, 60.0, 10)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    @patch("your_module_name.pd.read_csv")
    def test_compute_aep(self, mock_read_csv):
        # Mock dataset and power curve
        ds = xr.Dataset({
            "u10": (("time",), np.array([2.0, 3.0])),
            "v10": (("time",), np.array([2.0, 2.0])),
            "u100": (("time",), np.array([3.0, 4.0])),
            "v100": (("time",), np.array([3.0, 4.0]))
        }, coords={"time": pd.date_range("2023-01-01", periods=2), "latitude": 45.0, "longitude": -60.0})
        ds = ds.expand_dims(dim=["latitude", "longitude"])

        mock_df = pd.DataFrame({
            "Wind Speed": [0, 2, 4, 6, 8],
            "Power Output": [0, 100, 300, 600, 800]
        })
        mock_read_csv.return_value = mock_df

        result = mod.compute_aep(ds, 45.0, -60.0, "fake.csv", 2023, "NREL5MW")
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)


if __name__ == '__main__':
    unittest.main()
