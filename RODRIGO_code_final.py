"""
WIND RESOURCE ASSESSMENT
"""

# Imports
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from windrose import WindroseAxes
from scipy.stats import weibull_min
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import quad
from scipy.special import gamma

# --------------------------
# Data Loading and Preparation
# --------------------------

def load_wind_data(folder_path):
    """Load and combine NetCDF files from the specified folder."""
    datasets = []
    for file in Path(folder_path).iterdir():
        if file.suffix == '.nc':
            try:
                ds = xr.open_dataset(file)
                datasets.append(ds)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    if not datasets:
        raise ValueError("No valid NetCDF files found in the folder.")
    return xr.concat(datasets, dim='valid_time')

def process_wind_data(combined_ds):
    """Process wind data to calculate speed, direction, and organize by location."""
    WindData = combined_ds.to_dataframe().reset_index()
    WindData = WindData.rename(columns={'number': 'location'})
    WindData['location'] = list(zip(WindData['latitude'], WindData['longitude']))

    for height in [10, 100]:
        u_col, v_col = f'u{height}', f'v{height}'
        ws_col, wd_col = f'ws_{height}', f'wd_{height}'
        WindData[ws_col] = np.sqrt(WindData[u_col]**2 + WindData[v_col]**2)
        WindData[wd_col] = np.degrees(np.arctan2(WindData[u_col], WindData[v_col])) % 360

    locations = {
        f'Location_{i+1}': WindData[WindData['location'] == loc]
        for i, loc in enumerate([(7.75, 55.5), (8, 55.5), (7.75, 55.75), (8, 55.75)])
    }
    return WindData, locations

# --------------------------
# Plotting Functions
# --------------------------

def plot_wind_roses(locations):
    """Create 2x2 wind rose subplots."""
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': 'windrose'}, figsize=(12, 10))
    loc_names = ['Location 1 (7.75, 55.5)', 'Location 2 (8.0, 55.5)',
                 'Location 3 (7.75, 55.75)', 'Location 4 (8.0, 55.75)']
    
    for ax, (loc_name, data) in zip(axes.flat, locations.items()):
        ax.bar(data['wd_10'], data['ws_10'], normed=True, opening=0.8, edgecolor='white')
        ax.set_title(loc_names.pop(0), pad=20)
        ax.set_legend(title="Wind Speed (m/s)", loc='lower right', bbox_to_anchor=(1.1, -0.2))
    
    plt.tight_layout()
    plt.show()

def plot_weibull_distribution(wind_speeds, params, location_name, height):
    """Plot Weibull distribution and histogram."""
    shape, _, scale = params
    x = np.linspace(wind_speeds.min(), wind_speeds.max(), 100)
    pdf = weibull_min.pdf(x, *params)

    plt.figure(figsize=(10, 6))
    plt.hist(wind_speeds, bins=30, density=True, alpha=0.6, color='blue')
    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Weibull PDF (k={shape:.2f}, A={scale:.2f})')
    plt.title(f'Weibull Distribution - {location_name} at {height}m')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# --------------------------
# Weibull Analysis
# --------------------------

def fit_weibull(location_data, location_name):
    """Fit Weibull distribution and return mean wind speeds."""
    results = {}
    for height in [10, 100]:
        ws_col = f'ws_{height}'
        wind_speeds = location_data[ws_col]
        params = weibull_min.fit(wind_speeds, floc=0)
        mean_speed = params[2] * gamma(1 + 1/params[0])
        results[f'mean_{height}m'] = mean_speed
        plot_weibull_distribution(wind_speeds, params, location_name, height)
    return results

# --------------------------
# Wind Profile Calculations
# --------------------------

def calculate_power_law_profile(z, u_ref, z_ref, alpha=1/7):
    """Power law wind profile calculation."""
    return u_ref * (z/z_ref)**alpha

def calculate_logarithmic_profile(z, u_ref, z_ref, z0=0.0002):
    """Logarithmic law wind profile calculation."""
    return u_ref * (np.log(z/z0) / np.log(z_ref/z0))

def plot_wind_profiles(z_range, profiles, labels, title):
    """Plot wind profiles."""
    plt.figure(figsize=(10, 6))
    for profile, label in zip(profiles, labels):
        plt.plot(profile, z_range, label=label)
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Height (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# --------------------------
# Main Execution
# --------------------------

if __name__ == "__main__":
    # Load and process data
    folder_path = Path(r'inputs\WindData')
    combined_ds = load_wind_data(folder_path)
    WindData, locations = process_wind_data(combined_ds)

    # Plot wind roses
    plot_wind_roses(locations)

    # Perform Weibull analysis
    mean_speeds = []
    for loc_name, loc_data in locations.items():
        mean_speeds.append(fit_weibull(loc_data, loc_name))

    # Calculate and plot wind profiles
    z_range = np.linspace(0, 150, 300)
    for loc_data in mean_speeds:
        pl_100 = calculate_power_law_profile(z_range, loc_data['mean_100m'], 100)
        pl_10 = calculate_power_law_profile(z_range, loc_data['mean_10m'], 10)
        log_100 = calculate_logarithmic_profile(z_range, loc_data['mean_100m'], 100)
        log_10 = calculate_logarithmic_profile(z_range, loc_data['mean_10m'], 10)
        plot_wind_profiles(z_range, [pl_100, pl_10, log_100, log_10],
                           ['Power Law (100m)', 'Power Law (10m)', 'Log Law (100m)', 'Log Law (10m)'],
                           f'Wind Profiles - {loc_data["location"]}')
