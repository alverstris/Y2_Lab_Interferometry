from scipy.optimize import curve_fit
import numpy as np

def gaussian(x, a, x0, sigma, offset):
    """Gaussian function for curve fitting."""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def scan_function(x,y, threshold=0.15e-2):
    start_index, end_index = 0, 0
 
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
            
        

def fit_gaussian(x_data, y_data, p0=None):
    """Fit a Gaussian to the provided data."""
    if p0 is None:
        # sigma guess from variance
        mean = np.sum(x_data * y_data) / np.sum(y_data)
        sigma_guess = np.std(x_data, ddof = 1, mean = mean)

        a, x_max, x_min = max(y_data), max(x_data), min(x_data)
        p0 = [2*a, mean , sigma_guess, min(y_data)]
    
    lower_bounds = [0.2 * a, min(x_data), 0.01 * (x_max-x_min), -np.inf]
    upper_bounds = [np.inf, max(x_data), x_max-x_min, np.inf]
    bounds = (lower_bounds, upper_bounds)
    
    popt, pcov = curve_fit(gaussian, x_data, y_data, p0=p0, bounds=bounds)
    return popt, pcov