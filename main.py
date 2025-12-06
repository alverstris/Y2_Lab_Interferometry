import numpy as np
import matplotlib.pyplot as plt
from apply_global_calibration import fft_full_limited as fft 
from curvefitting import fit_gaussian, gaussian, scan_function

file = r"Data\green_1_white_2_8.8to13.8.txt"

x , y = fft(file)
x, y2 = fft(file, y_array_index_passed = 1)

def max_min_covariance_handler(components, covariance):
    std_devs = np.sqrt(np.diag(covariance))
    max, min = [], []
    for i in range(len(components)):
        max.append(components[i] + std_devs[i])
        min.append(components[i] - std_devs[i])
    return min, max

components, covariance = fit_gaussian(scan_function(x,y)[0], scan_function(x,y)[1])

# print(min(x))

min_vals, max_vals = max_min_covariance_handler(components, covariance)

plt.figure("Spectrum using global calibration FFT for green filter data")
plt.title('Data from: ' + file)
plt.plot(x,y2)
# plt.plot(x,y, label = "data_full", alpha = 1)
# plt.plot(scan_function(x,y)[0], scan_function(x,y)[1], label = "data_scanned",color = 'red', alpha = 1)

# plt.plot(scan_function(x,y)[0], gaussian(scan_function(x,y)[0], *components), label = "Gaussian fit", color = 'white' ,linestyle='--')
# plt.fill_between(scan_function(x,y)[0], gaussian(scan_function(x,y)[0], *min_vals), gaussian(scan_function(x,y)[0], *max_vals), color='blue', alpha=1, label='2$ \sigma $ range for fit')
plt.legend()
plt.xlim(0,10e-7)
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Wavelength (m)')
plt.grid()
plt.show()



