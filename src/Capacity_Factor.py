def compute_capacity_factor(aep, rated_power):
    """
    Computes the capacity factor of a wind turbine.

    Parameters:
    - aep (float): Annual Energy Production in GWh
    - rated_power (float): Turbine's rated power in MW

    Returns:
    - Capacity factor (float between 0 and 1)
    """
    max_possible_output = rated_power * 8760 / 1000  # Convert to GWh
    return aep / max_possible_output
