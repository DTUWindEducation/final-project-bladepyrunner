import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

class WindTurbine:
    """
    A class to represent a wind turbine and handle power curve data.
    
    Attributes:
        name (str): Turbine model name.
        hub_height (float): Hub height in meters.
        power_curve (pd.DataFrame): Power curve as a DataFrame with columns ['wind_speed', 'power'].
    """
    def __init__(self, name: str, hub_height: float, filepath: str):
        self.name = name
        self.hub_height = hub_height
        self.power_curve = pd.read_csv(filepath)
        self._interp_func = interp1d(self.power_curve['wind_speed'],
                                     self.power_curve['power'],
                                     bounds_error=False, fill_value=0.0)

    def get_power(self, wind_speeds: np.ndarray) -> np.ndarray:
        """
        Returns the interpolated power output for given wind speeds.
        """
        return self._interp_func(wind_speeds)

    def compute_AEP(self, wind_speed_pdf: callable, u_min: float, u_max: float, eta: float = 1.0) -> float:
        """
        Computes Annual Energy Production (AEP) using the provided wind speed probability density function.

        Args:
            wind_speed_pdf (callable): PDF function of wind speed (e.g., Weibull).
            u_min (float): Cut-in wind speed.
            u_max (float): Cut-out wind speed.
            eta (float): Turbine availability (default 1.0).

        Returns:
            float: Annual energy production in kWh.
        """
        def integrand(u):
            return self._interp_func(u) * wind_speed_pdf(u)
        
        energy = quad(integrand, u_min, u_max)[0]
        return eta * 8760 * energy  # hours in a year


#how to call it in the main code
# include this at the top of the file
# from src.turbine import WindTurbine


# somewhere in the main code where you want to use the WindTurbine class
# turbine_5MW = WindTurbine(
#     name="NREL 5 MW",
#     hub_height=90,
#     filepath="inputs/NREL_Reference_5MW_126.csv"
# )

# turbine_15MW = WindTurbine(
#     name="NREL 15 MW",
#     hub_height=150,
#     filepath="inputs/NREL_Reference_15MW_240.csv"
# )


# to test the class
# import numpy as np
# sample_wind_speeds = np.array([3, 5, 10, 15])
# power_output = turbine_5MW.get_power(sample_wind_speeds)
# print("Sample power output:", power_output)