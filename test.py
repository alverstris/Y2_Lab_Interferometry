# test.py


import numpy as np
import scipy.interpolate as spi
import numpy.fft as fft
import read_data_results3 as rd

def fft_spectra(file, N=100000, window=True):
    """
    Read file via rd.read_data3(file), build a uniformly-sampled OPD axis,
    perform FFT and return wavelength (m) and the one-sided magnitude spectrum.
    - Handles decreasing x by reversing arrays.
    - Enforces strict monotonic increase with minimal eps nudges (preserves data).
    - Removes DC after resampling and applies optional Hann window.
    """
    results = rd.read_data3(file)

    # scaling: microstep -> metres (use your value)
    metres_per_microstep = 15.6e-12
    metres_per_microstep *= 2.0   # mirror -> OPD conversion if needed

    y = np.array(results[0], dtype=float)
    x = np.array(results[5], dtype=float) * metres_per_microstep

    if len(x) < 3 or len(y) < 3:
        raise RuntimeError("Not enough data points")

    # If x is overall decreasing, reverse both arrays so x increases
    if np.mean(np.diff(x)) < 0:
        x = x[::-1]
        y = y[::-1]

    # Enforce strictly increasing x by minimal positive nudges (preserve data)
    diffs = np.diff(x)
    pos = diffs > 0
    if np.any(pos):
        median_forward = np.median(diffs[pos])
    else:
        # fallback: use median absolute diff
        median_forward = np.median(np.abs(diffs)) if diffs.size>0 else 1.0

    # ensure median_forward is a sensible positive number
    if not np.isfinite(median_forward) or median_forward <= 0:
        median_forward = max(np.abs(x).max() * 1e-12, 1e-12)

    eps = max(median_forward * 1e-6, 1e-15)  # tiny nudge

    # Apply minimal nudges in-place
    for i in range(1, len(x)):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + eps

    # Now resample with cubic spline onto a uniform grid
    # Defensive: require x to be strictly increasing now
    if not np.all(np.diff(x) > 0):
        raise RuntimeError("x is still not strictly increasing after nudging")

    xs = np.linspace(x[0], x[-1], N)
    cs = spi.CubicSpline(x, y)
    y_res = cs(xs)

    # Remove DC (important for magnitude-based spectra)
    y_res = y_res - np.mean(y_res)

    # Optional window to reduce spectral leakage
    if window:
        win = np.hanning(len(xs))
        y_win = y_res * win
    else:
        y_win = y_res

    # FFT and frequency axis (use correct spacing)
    dx = xs[1] - xs[0]
    Y = fft.fft(y_win)
    freqs = fft.fftfreq(N, d=dx)

    # positive frequencies only (skip zero-frequency/DC)
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    Y_pos = Y[pos_mask]

    # Convert spatial frequency (1/m) to wavelength (m)
    # filter out any extremely small freqs (defensive)
    if freqs_pos.size == 0:
        raise RuntimeError("No positive frequencies; check dx and N")

    wavelengths = 1.0 / freqs_pos

    spectrum = np.abs(Y_pos)

    return wavelengths, spectrum, (300e-9, 800e-9)
