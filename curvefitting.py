from scipy.optimize import curve_fit
import numpy as np

def gaussian(x, a, x0, sigma, offset):
    """Gaussian function for curve fitting."""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def fit_gaussian(x_data, y_data, p0=None):
    """Fit a Gaussian to the provided data."""
    if p0 is None:
        # sigma guess from variance
        sigma_guess = np.std(x_data)
        p0 = [max(y_data), x_data[np.argmax(y_data)], sigma_guess, min(y_data)]
    
    popt, pcov = curve_fit(gaussian, x_data, y_data, p0=p0)
    return popt, pcov