from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def sinc_squared(x,amp, omega,x_offset, y_offset):
    return amp * (np.sin(omega * x + x_offset)/(omega * x + x_offset))**2 + y_offset

def gaussian(x, a, x0, sigma, offset):
    """Gaussian function for curve fitting."""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def sinc_gaussian_combo(x,amp, omega, phi, x0, sigma, c):
    return amp * np.exp(-(x - x0)**2 / (2 * sigma**2)) * (np.sin(omega * x + phi)/(omega * x + phi))**2 + c

def scan_function(x,y, threshold=0.15e-2, threshold_or_range = True, window_range = (0, np.inf)):
    start_index, end_index = 0, 0

    # make sure x is going from min to max
    if np.argmax(x) < np.argmin(x):
        x = np.flip(x)
        y = np.flip(y)
    else:
        pass


    if threshold_or_range == True:
        
        y_max = max(y)

        y_mean_index = np.where(y == y_max)[0]

        lower_half, upper_half = np.arange(0, y_mean_index), np.arange(y_mean_index, len(x))

        for i in upper_half:
            if y[i] <= threshold * y_max:
                end_index = i
                break
        
        for j in reversed(lower_half):
            if y[j] <= threshold * y_max:
                start_index = j
                break
        
        if end_index - y_mean_index > y_mean_index - start_index:
            start_index = int(y_mean_index - (end_index - y_mean_index))
        else:
            end_index = int(y_mean_index + (y_max - start_index))

        
        return x[start_index:end_index], y[start_index:end_index]
    
    elif threshold_or_range == False:
       
        for i in range(len(x)):
            if x[i] >= window_range[0]:
                start_index = i
                break
            elif i == len(x) - 1:
                raise Exception("Did not start range")
        
        for i in range(start_index,len(x)):
            if x[i] >= window_range[1]:
                end_index = i
                break
            elif i == len(x) - 1:
                raise Exception("Did not end range")
        
        return x[start_index:end_index], y[start_index:end_index]
    else:
        raise Exception("Casting number to threshold_or_range, use True/False instead")
            

def fit_sinc(x,y, p0 = None):
    if p0 is None:
        rolling_data = np.zeros_like(y)
        for i in range(len(y)):
            rolling_data[i] = y[i]
        
        y_offset = (y[0]+y[-1])/2
        amp = max(y) - y_offset

        rolling_data -= y_offset
        central_peak_index = np.argmax(y)
        x_p = x[central_peak_index]
        for i in range(central_peak_index, len(x)):
            if rolling_data[i] <= 0:
                x_0 = x[i]
                break
        
        omega = np.pi/(x_0 - x_p)
        x_offset = - omega * x_p
        
        p0 = [amp, omega, x_offset, y_offset]
    
    plt.plot(x,sinc_squared(x, *p0))
    plt.plot(x,y)
    plt.xlim(100e-9,900e-9)
    plt.show()
    # lower_bounds = [1,                  1/min(x),      -np.inf,  0]
    # upper_bounds = [amp + 2 * y_offset, np.inf, 0,       amp + y_offset]
    # bounds = (lower_bounds, upper_bounds)
    # print()
    # print(p0)
    # print(lower_bounds)
    # print(upper_bounds)
    # print()
    popt, pcov = curve_fit(sinc_squared, x, y, p0=p0)
    return popt, pcov
    

def fit_gaussian(x_data, y_data, p0=None):
    """Fit a Gaussian to the provided data."""
    if p0 is None:
        # sigma guess from variance
        mean = np.sum(x_data * y_data) / np.sum(y_data)
        sigma_guess = np.std(x_data, ddof = 1, mean = mean)

        a, x_max, x_min = max(y_data), max(x_data), min(x_data)
        p0 = [2*a, mean , sigma_guess, min(y_data)]
    
    # lower_bounds = [0.2 * a, min(x_data), 0.01 * (x_max-x_min), -np.inf]
    # upper_bounds = [np.inf, max(x_data), x_max-x_min, np.inf]
    # bounds = (lower_bounds, upper_bounds)
    
    popt, pcov = curve_fit(gaussian, x_data, y_data, p0=p0)
    return popt, pcov

def sinc_combo(x,amp, omega, phi, omega_2, phi_2, c):
    return amp * (np.sin(omega * x + phi)/(omega* x + phi))**2 * np.sin(omega_2 * x + phi_2) + c

def fit_sinc_combo(x,y, p0=None):
    if p0 is None:
        rolling_data = np.zeros_like(y)
        for i in range(len(y)):
            rolling_data[i] = y[i]
        
        y_offset = (y[-1]+y[0])/2
        amp = max(y) - y_offset

        rolling_data -= y_offset
        central_peak_index = np.argmax(y)
        x_p = x[central_peak_index]
        for i in range(central_peak_index, len(x)):
            if rolling_data[i] <= 0:
                x_0 = x[i]
                break
        
        omega = np.pi/(x_0 - x_p)
        x_offset = - omega * x_p
        
        p0 =       [amp,    omega,  x_offset, omega * 3, x_offset, y_offset]
    
    upper_bounds = [np.inf, np.inf, np.inf,   np.inf,    np.inf,   np.inf ]
    lower_bounds = [0,      10,      -np.inf,  0,        -np.inf,   0]
    bounds = (lower_bounds, upper_bounds)
    popt, pcov = curve_fit(sinc_combo, x,y, p0=p0)
    return popt, pcov

    

# def combo_curve_fit(x,y):
#     # p0 = [16,0.5e-7, 0,  ,1]
#     gaussian_guess = fit_gaussian(x,y)[0][1:-1]

#     sinc_guess = fit_sinc(x,y)[0][:-1]
#     guess = np.concatenate((sinc_guess,gaussian_guess))
#     for i in guess:
#         p0.append(i)
#     p0.append(min(y))
#     print(p0)
#     print(len(p0))
#     popt, pcov = curve_fit(sinc_gaussian_combo, x, y, p0=p0)

#     return popt, pcov

