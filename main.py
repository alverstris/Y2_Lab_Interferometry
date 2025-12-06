import numpy as np
import matplotlib.pyplot as plt
from apply_global_calibration import fft_full_limited as fft 
from curvefitting import fit_gaussian, gaussian, scan_function

file = r"Data\green_1_white_2_8.8to13.8.txt"

x , y = fft(file)

components, covariance = fit_gaussian(scan_function(x,y)[0], scan_function(x,y)[1])

# print(min(x))

plt.figure("Spectrum using global calibration FFT for green filter data")
plt.title('Data from: ' + file)
# plt.plot(x,y, label = "data_full", alpha = 1)
plt.plot(scan_function(x,y)[0], scan_function(x,y)[1], label = "data_scanned", alpha = 1)

plt.plot(scan_function(x,y)[0], gaussian(scan_function(x,y)[0], *components), label = "Gaussian fit", linestyle='--')
plt.legend()
# plt.xlim(100e-8,600e-7)
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Wavelength (m)')
plt.grid()
plt.show()



