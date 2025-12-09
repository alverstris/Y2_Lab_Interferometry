import numpy as np

def parameter_consistency_test(popt_man, pcov_man, popt_meas, pcov_meas):
    """
    Computes z-scores comparing parameter sets from manufacturer vs. measurement.
    popt_* are parameter arrays, pcov_* are covariance matrices from curve_fit.
    """

    popt_man = np.asarray(popt_man)
    popt_meas = np.asarray(popt_meas)

    # Standard deviations (uncertainties) of the parameters
    sigma_man = np.sqrt(np.diag(pcov_man))
    sigma_meas = np.sqrt(np.diag(pcov_meas))

    # Combined uncertainty for each parameter
    sigma_combined = np.sqrt(sigma_man**2 + sigma_meas**2)

    # z-score for each parameter
    z_scores = (popt_man - popt_meas) / sigma_combined

    return z_scores, sigma_man, sigma_meas, sigma_combined


# Example usage:
# z, s_man, s_meas, s_comb = parameter_consistency_test(popt_man, pcov_man, popt_meas, pcov_meas)

# for i, z_i in enumerate(z):
#     print(f"Parameter {i}: z = {z_i:.2f}")
