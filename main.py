import numpy as np
import matplotlib.pyplot as plt
from apply_global_calibration import fft_full_limited as fft 
from curvefitting import scan_function, sinc_squared, fit_sinc

file = r"Data\green_1_white_2_8.8to13.8.txt"

x , y = fft(file)
x, y2 = fft(file, y_array_index_passed = 1)

def max_min_covariance_handler(components, covariance, value):
    std_devs = np.sqrt(np.diag(covariance))
    max, min = [], []
    for i in range(len(components)):
        max.append(components[i] + value * std_devs[i])
        min.append(components[i] - value * std_devs[i])
    return min, max




## for filter function now

# find ratio between values

ratios = np.zeros(len(y))
for i in range(len(y)):
    ratios[i] = y[i]/y2[i]

average_ratio = sum(ratios)/len(ratios)

y2_matched = y2 * average_ratio

# filter points

filter = np.zeros(len(y))
for i in range(len(y2_matched)):
    filter[i] = (y[i]/y2_matched[i])




scanned_data = scan_function(abs(x),abs(filter), threshold_or_range=False, window_range = (300e-9,600e-9))

components, covariance = fit_sinc(scanned_data[0], scanned_data[1])
# print(components)
plt.plot(x, sinc_squared(x, *components))

min_vals, max_vals = max_min_covariance_handler(components, covariance, 10e100)

print(min_vals, max_vals)
alpha_plots, alpha_region = 0.5, 1
plt.fill_between(x, sinc_squared(x, *min_vals), sinc_squared(x, *max_vals), color='grey', alpha=alpha_region, label='2$ \sigma $ range for fit')
plt.plot(x,abs(filter), label = "filter func data", alpha = alpha_plots)
# plt.plot(scanned_data[0],scanned_data[1], label = "shortened data", alpha = alpha_plots)
plt.plot(x, sinc_squared(x,*components), label = "fitted function", alpha = alpha_plots)

plt.title("Green filter as a STF")
plt.legend()
plt.xlim(100e-9,900e-9)
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Wavelength (m)')
plt.grid()
plt.show()



