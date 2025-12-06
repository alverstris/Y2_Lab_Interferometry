#!/usr/bin/python

import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi


# get the data and the x position
def fft_spectra(file):
    results = rd.read_data3(file)

    metres_per_microstep = 1.94e-11 # metres
    # if from Task 6, need to multiple by factor of 2 to account for the mirror movement to path difference conversion
    metres_per_microstep *= 2.0

    y1 = np.array(results[0])

    # get x-axis data from the results
    x = np.array(results[5])*metres_per_microstep

    # centre the y-axis on zero by either subtracting the mean

    y1 = y1 - y1.mean()

    # Cubic Spline part - the FFT requires a regular grid on the x-axis
    N = 100000 # these are the number of points that you will resample - try changing this and look how well the resampling follows the data.
    xs = np.linspace(x[0], x[-1], N) # x-axis to resample onto
    y = y1[:len(x)] # make sure y axis has same length as x 
    cs = spi.CubicSpline(x, y) # construct the cubic spline function


    # FFT to extract spectra
    yf1=spf.fft(cs(xs))
    xf1=spf.fftfreq(len(xs)) # setting the correct x-axis for the fourier transform. Osciallations/step  
    xf1=spf.fftshift(xf1) #shifts to make it easier (google if interested)
    yf1=spf.fftshift(yf1)
    xx1=xf1[int(len(xf1)/2+1):len(xf1)]

    # convert x-axis to meaningful units - wavelength
    distance = xs[1:]-xs[:-1]
    # rather than the amplitude
    repx1 = distance.mean()/xx1  

    # plt.figure("Spectrum using global calibration FFT")
    # plt.title('Data from: \n%s'%file)
    # plt.plot(abs(repx1),abs(yf1[int(len(xf1)/2+1):len(xf1)]))
    # plt.xlim(300e-9,800e-9)
    # plt.ylabel('Intensity (a.u.)')
    # plt.show()

    return abs(repx1),abs(yf1[int(len(xf1)/2+1):len(xf1)]),(300e-9,800e-9)




