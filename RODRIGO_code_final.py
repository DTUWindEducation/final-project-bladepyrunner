# %%
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.stats import weibull_min
from scipy.special import gamma
from scipy.integrate import quad
from windrose import WindroseAxes

# Configuration Constants
CONFIG = {
    "data_paths": {
        "wind_data": Path('inputs/WindData'),
        "nrel_15mw": Path('inputs/NREL_Reference_15MW_240.csv'),
        "nrel_5mw": Path('inputs/NREL_Reference_5MW_126.csv')
    },
    "locations": {
        "Location 1": (7.75, 55.5),
        "Location 2": (8.0, 55.5),
        "Location 3": (7.75, 55.75),
        "Location 4": (8.0, 55.75)
    },
    "analysis_params": {
        "target_point": (7.90, 55.60),
        "z_refs": (10, 100),
        "z_target": 70,
        "alpha": 1/7,
        "z0": 0.0002,
        "loss_factor": 0.15
    }
}

# %%
def load_and_process_data():
    """Load and process wind data from NetCDF files"""
    datasets = []
    for file in CONFIG["data_paths"]["wind_data"].iterdir():
        datasets.append(xr.open_dataset(file))
    
    combined_ds = xr.concat(datasets, dim='valid_time')
    df = combined_ds.to_dataframe().reset_index()
    
    # Process wind components
    for height in [10, 100]:
        u_col = f'u{height}'
        v_col = f'v{height}'
        df[f'ws_{height}'] = np.hypot(df[u_col], df[v_col])
        df[f'wd_{height}'] = np.degrees(np.arctan2(df[u_col], df[v_col])) % 360
    
    df['location'] = list(zip(df['latitude'], df['longitude']))
    return df

def create_location_datasets(df):
    """Create separate datasets for each location"""
    return {name: df[df["location"] == coords] 
            for name, coords in CONFIG["locations"].items()}

def plot_wind_roses(location_data):
    """Create wind rose plots for all locations"""
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': 'windrose'}, 
                            figsize=(12, 10))
    
    for ax, (loc_name, data) in zip(axes.flat, location_data.items()):
        ax.bar(data['wd_10'], data['ws_10'], 
              normed=True, opening=0.8, edgecolor='white')
        ax.set_title(loc_name)
        ax.set_legend(title="Wind Speed (m/s)", loc='lower right')
    
    plt.tight_layout()
    plt.show()

def weibull_analysis(location_data):
    """Perform Weibull analysis and return results"""
    results = []
    
    def analyze_location(data, loc_name):
        """Analyze single location"""
        params = {}
        for height in [10, 100]:
            ws = data[f'ws_{height}']
            shape, loc, scale = weibull_min.fit(ws, floc=0)
            mean = scale * gamma(1 + 1/shape)
            
            # Plotting
            x = np.linspace(ws.min(), ws.max(), 100)
            pdf = weibull_min.pdf(x, shape, loc, scale)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(ws, bins=30, density=True, alpha=0.6, 
                    color='blue' if height == 10 else 'green')
            ax.plot(x, pdf, 'r-', label=f'k={shape:.2f}, A={scale:.2f}\nMean: {mean:.2f} m/s')
            ax.set_title(f'{loc_name} at {height}m')
            ax.legend()
            plt.show()
            
            params[f'ws{height}'] = {'shape': shape, 'scale': scale, 'mean': mean}
        
        return {loc_name: params}
    
    for loc_name, data in location_data.items():
        results.append(analyze_location(data, loc_name))
    
    return results

def create_interpolators(location_data):
    """Create spatial interpolators for wind data"""
    points = np.array(list(CONFIG["locations"].values()))
    x_coords = np.unique(points[:, 0])
    y_coords = np.unique(points[:, 1])
    
    def prepare_grid(data, height):
        """Prepare data grid for interpolation"""
        arr = np.array([loc[f'ws_{height}'].values for loc in location_data.values()])
        return arr.reshape(2, 2, -1)
    
    return {
        'ws10': prepare_grid(location_data, 10),
        'ws100': prepare_grid(location_data, 100),
        'coords': (x_coords, y_coords)
    }

def calculate_wind_profiles(weibull_results):
    """Calculate and plot wind profiles"""
    z = np.linspace(0, 200, 400)
    results = []
    
    def plot_profiles(loc_name, power, log):
        """Plot profiles for single location"""
        plt.figure(figsize=(10, 6))
        plt.plot(power, z, label='Power Law')
        plt.plot(log, z, '--', label='Logarithmic Law')
        plt.title(f'Wind Profile - {loc_name}')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Height (m)')
        plt.legend()
        plt.grid()
        plt.show()
    
    for loc_data in weibull_results:
        loc_name = list(loc_data.keys())[0]
        params = list(loc_data.values())[0]
        
        # Power Law calculation
        power = params['ws100']['mean'] * (z/CONFIG["analysis_params"]["z_refs"][1])**CONFIG["analysis_params"]["alpha"]
        
        # Logarithmic Law calculation
        log = params['ws100']['mean'] * (np.log(z/CONFIG["analysis_params"]["z0"]) / 
              np.log(CONFIG["analysis_params"]["z_refs"][1]/CONFIG["analysis_params"]["z0"]))
        
        plot_profiles(loc_name, power, log)
        results.append({loc_name: {'power_law': power, 'log_law': log}})
    
    return results

def calculate_aep(wind_params, turbine_specs):
    """Calculate Annual Energy Production"""
    results = {}
    
    # Load turbine power curves
    turbines = {
        name: interp1d(pd.read_csv(spec['path'])['Wind Speed [m/s]'].values,
                        pd.read_csv(spec['path'])['Power [kW]'].values, 
                        bounds_error=False, fill_value=0)
        for name, spec in turbine_specs.items()
    }
    
    def integrand(u, power_curve, k, A):
        return power_curve(u) * weibull_min.pdf(u, k, scale=A)
    
    for name, curve in turbines.items():
        integral, _ = quad(integrand, 
                          turbine_specs[name]['cut_in'], 
                          turbine_specs[name]['cut_out'],
                          args=(curve, wind_params['k'], wind_params['A']))
        aep = (1 - CONFIG["analysis_params"]["loss_factor"]) * 8760 * integral / 1e6
        results[name] = aep
    
    return results

# %%
# Main Analysis Pipeline
if __name__ == "__main__":
    # Load and process data
    wind_data = load_and_process_data()
    locations = create_location_datasets(wind_data)
    
    # Wind rose analysis
    plot_wind_roses(locations)
    
    # Weibull analysis
    weibull_results = weibull_analysis(locations)
    
    # Wind profile analysis
    profile_results = calculate_wind_profiles(weibull_results)
    
    # Spatial interpolation
    interpolators = create_interpolators(locations)
    
    # AEP calculation
    turbine_specs = {
        "NREL_15MW": {"path": CONFIG["data_paths"]["nrel_15mw"], "cut_in": 3.0, "cut_out": 25.0},
        "NREL_5MW": {"path": CONFIG["data_paths"]["nrel_5mw"], "cut_in": 3.0, "cut_out": 25.0}
    }
    
    # Fit Weibull parameters for target location
    k, _, A = weibull_min.fit(interpolators['ws100'][...], floc=0)  # Simplified for example
    aep_results = calculate_aep({'k': k, 'A': A}, turbine_specs)
    
    # Final output
    print("\n=== Final Results ===")
    print(f"{'Turbine Model':<20} {'AEP (GWh/year)':>15}")
    for name, value in aep_results.items():
        print(f"{name:<20} {value:>15.2f}")
    
    print("\n=== Weibull Parameters at Target Location ===")
    print(f"{'Shape (k):':<20} {k:.2f}")
    print(f"{'Scale (A):':<20} {A:.2f}")
    print(f"{'Mean Speed:':<20} {A * gamma(1 + 1/k):.2f} m/s")