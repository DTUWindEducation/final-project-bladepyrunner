import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import weibull_min
from scipy.integrate import quad
from src import load_nc_files, compute_speed_direction, interpolate_wind_data, power_law_calculation, fit_weibull, plot_weibull, plot_windrose, compute_aep, compute_capacity_factor, compute_mean_wind_speed
from src.turbine import WindTurbine
import sys
from pathlib import Path

# Add the project root (one level up from main.py) to the system path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Step 1: Load the wind data from the NetCDF files
print('Loading the wind data...')

data_dir = Path('inputs/WindData')
ds = load_nc_files(data_dir)

u10 = ds['u10'].values
clean_u10 = u10[~np.isnan(u10)]
v10 = ds['v10'].values
clean_v10 = v10[~np.isnan(v10)]
u100 = ds['u100'].values
clean_u100 = u100[~np.isnan(u100)]
v100 = ds['v100'].values
clean_v100 = v100[~np.isnan(v100)]

# Step 2: Calculate the wind speed and direction for all locations
print('Calculating wind speed and direction...')
wind_speed_10, wind_dir_10 = compute_speed_direction(clean_u10, clean_v10)
wind_speed_100, wind_dir_100 = compute_speed_direction(clean_u100, clean_v100)

# Step 3: Finding Wind Speeds inside the target area using interpolation
print('Interpolating wind speeds to the target area...')
interpolated_data = interpolate_wind_data(ds, target_lat=55.5, target_lon=7.8)
# Access the variables
print(f'Wind speed at 10m: {interpolated_data["wind_speed_10"].values}')
print(f'Wind direction at 10m: {interpolated_data["wind_dir_10"].values}')
print(f'Wind speed at 100m: {interpolated_data["wind_speed_100"].values}')
print(f'Wind direction at 100m: {interpolated_data["wind_dir_100"].values}')

# Step 4. Calculating wind speeds at different heights using the power law profile
print('Calculating wind speeds at different heights using the power law profile...')
z_ref = 10  # Reference height (m)
u_ref = wind_speed_10  # Wind speed at reference height (m/s)
z = 80  # Height at which to calculate wind speed (m)
u_80 = power_law_calculation(z_ref, u_ref, z)

# Step 5: Fitting the Weibull distribution to the wind speed data
print('Fitting the Weibull distribution to the wind speed data...')
shape, loc, scale = fit_weibull(ds, 55.5, 8, 10)
print(f"Weibull parameters: shape={shape}, loc={loc}, scale={scale}")
# Define wind speed PDF using Weibull distribution
def weibull_pdf(u):
    return weibull_min.pdf(u, shape, loc, scale)
# Instantiate the WindTurbine object
turbine_5MW = WindTurbine(
    name="NREL 5 MW",
    hub_height=90,
    filepath="inputs/power_curve/NREL_Reference_5MW_126.csv"
)

# Step 6: Plotting the Weibull distribution
print('Plotting the Weibull distribution...')
plot_weibull(ds, 55.5, 8, height=10)

# Step 7: Plotting the wind rose
print('Plotting the wind rose...')
plot_windrose(ds, 55.5, 8, height=10)

# Step 8: Calculating the AEP
print('Calculating the AEP...')
aep = turbine_5MW.compute_AEP(wind_speed_pdf=weibull_pdf, u_min=3, u_max=25, eta=0.95)
aep = aep / 1e6  # convert Wh to GWh
print(f"Annual Energy Production (AEP): {aep:.2f} GWh")

# Additional Function 1: Compute Capacity Factor
print('Calculating the Capacity Factor...')
rated_power = 5  # MW
capacity_factor = compute_capacity_factor(aep, rated_power)
print(f"Capacity Factor: {capacity_factor:.2%}")

# Additional Function 2: Compute Mean Wind Speeds
print('Calculating the Mean Wind Speeds...')
mean_wind_speed_10 = compute_mean_wind_speed(ds, 55.5, 8, 10)
print(f"Mean Wind Speed at 10m: {mean_wind_speed_10:.2f} m/s")


# Create a dictionary of results
results = {
    'Weibull Shape': [shape],
    'Weibull Loc': [loc],
    'Weibull Scale': [scale],
    'AEP (GWh)': [aep],
    'Capacity Factor': [capacity_factor],
    'Mean Wind Speed at 10m (m/s)': [mean_wind_speed_10],
    'Interpolated Wind Speed at 10m (m/s)': [interpolated_data["wind_speed_10"].values],
    'Interpolated Wind Direction at 10m (degrees)': [interpolated_data["wind_dir_10"].values],
    'Interpolated Wind Speed at 100m (m/s)': [interpolated_data["wind_speed_100"].values],
    'Interpolated Wind Direction at 100m (degrees)': [interpolated_data["wind_dir_100"].values],
}

# Convert the dictionary to a pandas DataFrame
df_results = pd.DataFrame(results)

# Define the output file path
output_file = Path('outputs/wind_resource_assessment_results.csv')

# Save the results to a CSV file
df_results.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")