import numpy as np
import matplotlib.pyplot as plt
from apply_global_calibration import fft_full_limited as fft 
from curvefitting import scan_function, gaussian, fit_gaussian
from analysis import parameter_consistency_test

file = r"Data\green_1_white_2_8.8to13.8.txt"

x , y = fft(file)

# plt.plot(x,abs(y))
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Intensity (%)")
# plt.title("Green Spectrum (w/o deconvolution)")
# plt.xlim(300e-9, 900e-9)
# plt.grid()
# plt.show()

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

# ratios = np.zeros(len(y))
# for i in range(len(y)):
#     ratios[i] = y[i]/y2[i]

# average_ratio = sum(ratios)/len(ratios)

# y2_matched = y2 * average_ratio

# filter points

filter = np.zeros(len(y))
for i in range(len(y2)):
    filter[i] = (y[i]/y2[i])

normaliser = max(filter)
for o in range(len(filter)):
    filter[o] /= normaliser
    



scanned_data = scan_function(abs(x),abs(filter), threshold_or_range=False, window_range = (350e-9,750e-9))


# sigma gen
rolling_mean = []
for i in scanned_data[1]:
    if i < 0.2:
        rolling_mean.append(i)

true_mean = sum(rolling_mean) / len(rolling_mean)

yerr = np.zeros_like(scanned_data[1])
for i in range(len(yerr)):
    # if scanned_data[1][i] >= 0.2:
    #     yerr[i] = true_mean
    # else:
    yerr[i] = true_mean / scanned_data[1][i]


components, covariance = fit_gaussian(scanned_data[0], scanned_data[1], sigma = yerr, absolute_sigma = True)
# # print(components)
# plt.plot(x, gaussian(x, *components))

# min_vals, max_vals = max_min_covariance_handler(components, covariance, 4)


x_lim_region_Viewport = (200e-9,900e-9)

points_per_section = 10e3 #section is 100e-9 long

# print(min_vals, max_vals)
# alpha_plots, alpha_region = 0.5, 1


# plt.fill_between(x, gaussian(x, *min_vals), gaussian(x, *max_vals), color='grey', alpha=alpha_region, label='2$ \sigma $ range for fit')


# plt.plot(x,abs(filter), label = "filter func data normalised", alpha = alpha_plots)
# plt.plot(scanned_data[0],scanned_data[1], label = "shortened data", alpha = alpha_plots)

x_axis_res_increase = np.linspace(x_lim_region_Viewport[0], x_lim_region_Viewport[1], int((x_lim_region_Viewport[1] - x_lim_region_Viewport[0])/100e-9 * points_per_section))

# plt.plot(x_axis_res_increase, gaussian(x_axis_res_increase,*components), label = "fitted function", alpha = alpha_plots)


# plt.plot(x,abs(y2), label= "normalised data (white)")
# plt.title("Data from: " + file)
# plt.title("Green filter as a STF")
# plt.legend()
# plt.xlim(x_lim_region_Viewport)
# plt.ylabel('Intensity (a.u.)')
# plt.xlabel('Wavelength (m)')
# plt.grid()
# plt.show()


components_data, covariance_data = fit_gaussian(scanned_data[0], scanned_data[1], sigma = yerr, absolute_sigma = True)

manufacturer_file = r"data_2.csv"

manufacturer_wav, manufacturer_int = np.loadtxt(manufacturer_file, delimiter = ",", skiprows = 1, unpack= True)

min_wav, max_wav, min_int, max_int = min(manufacturer_wav), max(manufacturer_wav), min(manufacturer_int), max(manufacturer_int)

manufacturer_wav /= 10e8
manufacturer_int /= max_int

# extend manufacturer wav range to 350 by 750 to match other

low_no_points, high_no_points = 0,0
check = False



for i in range(len(scanned_data[0])):
    if scanned_data[0][i] >= min_wav and check == False:
        low_no_points = i
        check = True
    elif scanned_data[0][i] >= max_wav:
        high_no_points = i
        break
    
low_points, high_points = np.zeros(low_no_points), np.zeros(high_no_points)

low_points.fill(min_int)
high_points.fill(min_int)

full_spectrum_manufacturer_int = np.concatenate((low_points, manufacturer_int, high_points))

low_points_wav, high_points_wav = np.linspace(350e-9, min_wav, low_no_points), np.linspace(max_wav, 750e-9, high_no_points)

full_spectrum_manufacturer_wav = np.concatenate((low_points_wav, manufacturer_wav, high_points_wav))



components_manufacturer, covariance_manufacturer = fit_gaussian(full_spectrum_manufacturer_wav, full_spectrum_manufacturer_int)


# plt.plot(full_spectrum_manufacturer_wav, gaussian(full_spectrum_manufacturer_wav, *components), label = "Experimental func fit")
# plt.plot(full_spectrum_manufacturer_wav, gaussian(full_spectrum_manufacturer_wav, *components_manufacturer), label = "Manufacturer func fit")
# plt.grid()
# plt.legend()
# plt.title("Fitted Functions Side by Side")
# plt.show()


def cov_handler(covariance):
    std_devs = np.sqrt(np.diag(covariance))
    return std_devs

print(cov_handler(covariance_data))
print()
print(cov_handler(covariance_manufacturer))

# stats = parameter_consistency_test(components_manufacturer, covariance_manufacturer, components_data, covariance_data)

# print(stats[0])