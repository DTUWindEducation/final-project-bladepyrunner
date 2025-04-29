import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import weibull_min

def plot_weibuill(wind_speed, k, A):
    x= np.linspace(0, max(wind_speed), 100)
    plt.hist(wind_speed, bins=30, density=True, alpha=0.6, color='g', label='Data')
    plt.plot(x, weibull_min.pdf(x, k, scale=A), 'r-', lw=2, label='Weibull PDF')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.show()