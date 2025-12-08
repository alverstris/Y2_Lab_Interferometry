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

file = "data_2.csv"
wavelengths, intensity = np.loadtxt(file, skiprows=1, delimiter=",", unpack=True)

components, covariance = fit_gaussian(wavelengths, intensity)
residuals = intensity - gaussian(wavelengths, *components)
degrees_of_freedom = len(intensity) - len(components)
sigma_estimated = np.sqrt(np.sum(residuals**2)/degrees_of_freedom)

yerr = np.full_like(intensity, sigma_estimated)

components_new,covariance = fit_gaussian(wavelengths,intensity, p0 = components, sigma = yerr, absolute_sigma=True)

print(ch(wavelengths, intensity,min(intensity) ,gaussian, components_new))


min_vals, max_vals = max_min_covariance_handler(components_new, covariance, 4)
alpha_plots, alpha_region = 0.5, 1

plt.plot(wavelengths,intensity, label = "data", alpha = alpha_plots)
plt.plot(wavelengths, gaussian(wavelengths, *components_new), label = "fit", alpha = alpha_plots)
plt.fill_between(wavelengths, gaussian(wavelengths, *min_vals), gaussian(wavelengths, *max_vals), color='grey', label='2$\sigma$ range for fit', alpha = alpha_region)
plt.grid()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (%)")
plt.title("Fitted Gaussian onto Manufacturer Data")
plt.legend()
plt.show()



