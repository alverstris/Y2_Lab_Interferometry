import numpy as np
import matplotlib.pyplot as plt
from apply_global_calibration import fft_spectra as fft_global

file = r"Data/green_1_white_2_8.8to13.8.txt"

x , y , xlim = fft_global(file)

plt.title('FFT Spectrum using Global Calibration')
plt.plot(x, y, label = str(file))
plt.xlim(xlim)
plt.xlabel('Wavelength (m)')
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.show()

