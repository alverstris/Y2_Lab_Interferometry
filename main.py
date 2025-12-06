import numpy as np
import matplotlib.pyplot as plt
from apply_global_calibration import fft_full_limited as fft 
from curvefitting import fit_gaussian, gaussian

file = r"Data\green_1_white_2_8.8to13.8.txt"

x , y = fft(file)

components, covariance = fit_gaussian(x, y)


plt.figure("Spectrum using global calibration FFT for green filter data")
plt.title('Data from: ' + file)
plt.plot(x, y, label = "data")
plt.plot(x, gaussian(x, *components), label = "Gaussian fit", linestyle='--')
plt.legend()
plt.xlim(300e-9,600e-9)
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Wavelength (m)')
plt.grid()
plt.show()



