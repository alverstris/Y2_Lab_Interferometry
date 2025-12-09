import matplotlib.pyplot as plt
import numpy as np
from curvefitting import gaussian, fit_gaussian
from chi_squared_test import chi_squared_test as ch

def max_min_covariance_handler(components, covariance, value):
    std_devs = np.sqrt(np.diag(covariance))
    max, min = [], []
    for i in range(len(components)):
        max.append(components[i] + value * std_devs[i])
        min.append(components[i] - value * std_devs[i])
    return min, max

def window_smoothing_function(x,y, window = 3):
    rolling_x, rolling_y = 0, 0
    new_x, new_y = [], []
    for i in range(len(x)):
        rolling_x += x[i]
        rolling_y += y[i]
        if (i+1) % window == 0:
            new_x.append(rolling_x/window)
            new_y.append(rolling_y/window)
            rolling_x, rolling_y = 0, 0
    return new_x, new_y
        


    


file = "data_2.csv"
wavelengths, intensity = np.loadtxt(file, skiprows=1, delimiter=",", unpack=True)





# print(ch(wavelengths, intensity,min(intensity) ,gaussian, components))


# min_vals, max_vals = max_min_covariance_handler(components, covariance, 4)
alpha_plots, alpha_region = 0.5, 1

sim_wave, sim_int = window_smoothing_function(wavelengths,intensity, window = 5)


components,covariance = fit_gaussian(sim_wave,sim_int, maxfev=10e6)

plt.plot(sim_wave,sim_int, label = "data", alpha = alpha_plots)
plt.plot(wavelengths, gaussian(wavelengths, *components), label = "fit", alpha = alpha_plots)
# plt.fill_between(wavelengths, gaussian(wavelengths, *min_vals), gaussian(wavelengths, *max_vals), color='grey', label='2$\sigma$ range for fit', alpha = alpha_region)
plt.grid()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (%)")
plt.title("Fitted Gaussian onto Manufacturer Data")
plt.legend()
plt.show()



