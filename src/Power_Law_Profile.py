def power_law_calculation(z_ref, u_ref, z, alpha=1/7):
    """
    Calculate wind speed at height z using the power law profile.
    Parameters:
    z_ref : float
        Reference height (m)
    u_ref : float
        Wind speed at reference height (m/s)
    z : float
        Height at which to calculate wind speed (m)
    alpha : float
        Power law exponent (dimensionless)
    Returns : float
        Wind speed at height z (m/s)
    """
    # Check if the inputs are valid
    return u_ref * (z / z_ref) ** alpha
