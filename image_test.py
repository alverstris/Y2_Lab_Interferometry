import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
file = "data.csv"
file_filtered = "data_2.csv"

data = np.loadtxt("data.csv", delimiter=',', skiprows=1)

wavelength, intensity = data[:,0], data[:,1]

offset = - 2* min(intensity)
intensity += offset

df = {"wavelength (nm)" : wavelength,
      "Intensity (%)" : intensity}

df = pd.DataFrame(df)

df.to_csv(file_filtered, index=False)

# plt.plot(wavelength,intensity)
# plt.grid()
# plt.xlabel("wavelength (nm)")
# plt.ylabel("intensity %")
# plt.title("Manufacturer filter data")
# plt.show()