import numpy as np
from scipy.stats import chi2

def chi_squared_test(x, y, yerr, model, popt):
    """
    Computes chi-square, reduced chi-square, and p-value.
    
    Parameters
    ----------
    x : array
        x data
    y : array
        y data
    yerr : array
        uncertainties in y
    model : callable
        function f(x, *params)
    popt : array
        best-fit parameters from curve_fit
        
    Returns
    -------
    chi2_value : float
    red_chi2 : float
    dof : int
    p_value : float
    """
    
    # residuals
    residuals = y - model(x, *popt)
    
    # chi-square
    chi2_value = np.sum((residuals / yerr)**2)
    
    # degrees of freedom
    dof = len(x) - len(popt)
    
    # reduced chi-square
    red_chi2 = chi2_value / dof
    
    # p-value (probability of getting chi2 >= this if model is correct)
    p_value = 1 - chi2.cdf(chi2_value, df=dof)
    
    return chi2_value, red_chi2, dof, p_value