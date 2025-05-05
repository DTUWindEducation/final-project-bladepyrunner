class CapacityFactorCalculator:
    """
    Class to compute capacity factor given AEP and rated power.
    """

    def __init__(self, rated_power_mw):
        """
        Parameters:
            rated_power_mw (float): Turbine rated power in megawatts (MW)
        """
        self.rated_power_mw = rated_power_mw

    def compute(self, aep_gwh):
        """
        Compute the capacity factor.

        Parameters:
            aep_gwh (float): Annual energy production in GWh

        Returns:
            float: Capacity factor (0–1)
        """
        max_possible_energy = self.rated_power_mw * 8760 / 1000  # convert MWh to GWh
        return aep_gwh / max_possible_energy