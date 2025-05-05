**Team**: BladePYrunners 

## Overview

**bladepyrunner** is a Python package designed to analyze wind resource data and estimate the performance of wind turbines. Using wind datasets in NetCDF or CSV format, the package computes wind speed distributions, interpolates wind vectors to geographic coordinates, and estimates turbine performance metrics such as:

- Annual Energy Production (AEP)
- Capacity Factor
- Mean Wind Speed

It supports interpolation at custom hub heights and includes preconfigured turbine models like the NREL 5MW and 15MW.

---

## Quick-start guide

To install and run the package locally:

```bash
git clone https://github.com/your-username/bladepyrunner.git
cd bladepyrunner
``` 
Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
Install the package in editable mode:
```bash
pip install -e .
```
Run the example code:
```bash
python -m examples.main
```

## Architecture

```bash
bladepyrunner/
├── inputs/
│   ├── NREL_Reference_5MW_126.csv
│   ├── NREL_Reference_15MW_240.csv
│   └── WindData/
├── outputs/              # Generated figures and outputs
├── src/                  # Core package code
│   ├── __init__.py
│   └── turbine.py
├── tests/
├── examples/             # Usage demos
│   ├── main.py
│   ├── FINAL_CODE.py
│   ├── main_code.ipynb
│   └── wind_data_class.py
├── .gitignore
├── LICENSE
├── Collaboration.md
├── README.md
└── pyproject.toml
```

### High-Level Workflow
        +---------------------+
        |  Wind Data (NetCDF) |
        +---------+-----------+
                  |
                  v
        +---------+-----------+
        | load_nc_files()     |
        | interpolate_wind... |
        +---------+-----------+
                  |
                  v
     +------------+-------------+
     |  Wind speed & direction |
     +------------+-------------+
                  |
         +--------+--------+
         |  Analysis Tools |
         |  (AEP, Weibull, |
         |   windrose...)  |
         +--------+--------+
                  |
        +---------+---------+
        | Visualization &   |
        | Summary Metrics   |
        +-------------------+


## Classes and Key Files

### `turbine.py` (in `src/`)

Defines the `WindTurbine` class, which models a turbine using its power curve and enables AEP (Annual Energy Production) computation from wind data.

#### `WindTurbine` class

- **Constructor**  
  `WindTurbine(name, hub_height, filepath)`  
  Initializes the turbine with:
  - `name` (str): Turbine model name  
  - `hub_height` (float): Hub height in meters  
  - `filepath` (str): Path to a CSV file with columns `wind_speed` and `power`

- **`get_power(wind_speeds)`**  
  Returns the interpolated power output (in kW) for a NumPy array of wind speeds.

- **`compute_AEP(wind_speed_pdf, u_min, u_max, eta=1.0)`**  
  Computes the **Annual Energy Production (AEP)** using a wind speed probability density function (PDF).
  - `u_min`, `u_max`: Cut-in and cut-out wind speeds  
  - `eta`: Turbine availability (default = 1.0)  
  Returns AEP in **kWh/year**.

---

### Functions in `__init__.py` (in `src/`)

Core functions used across the analysis pipeline:

- `load_nc_files()` – Load NetCDF climate data  
- `interpolate_wind_data()` – Interpolate wind speeds at a given hub height  
- `compute_speed_direction()` – Calculate wind speed and direction from U/V components  
- `fit_weibull()` / `plot_weibull()` – Fit and visualize a Weibull distribution from wind data  
- `compute_aep()` / `compute_capacity_factor()` – Estimate energy production and efficiency  
- `plot_windrose()` – Generate wind rose visualizations of wind patterns

## Team contributions

- Rodrigo Sanchez Moreno
- Max Rosendahl
- Cristina Fente Gutierrez