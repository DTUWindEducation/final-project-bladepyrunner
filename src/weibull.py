from scipy.stats import weibull_min

def fit_weibull(wind_speed):
    """
    Calculate the Weibull parameters (scale and shape) for a given wind speed data.
    
    Parameters:
    Wind_speed : array-like
        Wind speed data (m/s)
    
    returns:
    tuple : (scale, shape)
        Weibull parameters (scale, shape)
    """
    # Fit the Weibull distribution to the wind speed data
    shape, loc, scale = weibull_min.fit(wind_speed, floc=0)
    
    return scale, shape, loc