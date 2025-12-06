import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



def rd(fname):
    ######################################################################
    # A function that reads in the file. Modified from DJC 1/10/2018 by AC 4/03/20
    # Calling arguements:
    #     fname = name of txt file 
    #A better version of read_data2 that extract info out of a text file of any length i with result : [[0], [1], ...... [i]] where [i] is the array containing all the info of the ith column of the txt, all numbers are float, carefull when we have a single column: [ [0] ] array within array !
    
    #3rd - modif to store file names and not floats anymore - used to sort out the export procedure
    ######################################################################
    file = open(fname,"r")#, encoding='mac_roman')
    
    lines= file.readlines()
    signal=[[]]
    i=0
    j=0
    for k in lines[0]:
        #print (k)
        if k == ' ':
            signal.append([])
    #print(signal)
    for line in range(1,len(lines)):
        #print(line,'line')
        j=0
        i=0
        while j<=len(lines[line]) and i<=len(lines[line])-2:   #careful the 2 has been changed here, it was a 1 before !
            dd='' 
            #print(i,j)
            #print(signal)
            while lines[line][i]!=' ' and i<=len(lines[line])-2:
                dd=dd+lines[line][i]
                i=i+1
            
            #print(j)
            #print(dd)
            signal[j].append(float(dd))
            j=j+1
            i=i+1
            
            
    file.close()
    return signal

# Read in the data

def main(file):
    #this is the data
    results = rd(file)
    x = np.array(results[5])
    y = np.array(results[0])

    # Use a Butterworth filter to correct offset
    filter_order = 2  # type of filter
    freq = 1
    sampling_freq = 50
    # replace original array with new filtered one
    y = signal.sosfilt(signal.butter(filter_order, freq, 'hp', fs=sampling_freq,
                                    output='sos'), y)

    # Removes artefacts introduced by the butterworth filter
    percent_mask = 0.05  # you may be able to make this smaller
    y = y[int(len(y)*(percent_mask)):int(len(y)*(1-percent_mask))]
    x = x[int(len(x)*(percent_mask)):int(len(x)*(1-percent_mask))]
    x -= min(x)  # resets the x values to be centred about 0

    # Get the x points at which we cross the x - axis
    crossing_points = np.array([])
    for i in range(len(y)-1):
        # go through, check for sign change in y
        if (y[i] <= 0 and y[i+1] >= 0) or (y[i] >= 0 and y[i+1] <= 0):
            # create the exact crossing point of 0
            b = (y[i+1] - y[i]/x[i] * x[i+1])/(1-x[i+1]/x[i])
            a = (y[i] - b)/x[i]
            extra = -b/a - x[i]
            crossing_points = np.append(crossing_points, (x[i]+extra))

    # Finds the distance between each crossing point
    diff = np.array([])
    for i in range(len(crossing_points)-1):
        diff = np.append(diff, (np.abs(crossing_points[i+1]-crossing_points[i])))

    # Plot Results
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))
    fig.suptitle('Crossing Point Analysis', fontsize=20)

    ax[0].plot(x, y, 'x-', label='Data', color='blue')
    ax[0].plot(crossing_points, 0*np.array(crossing_points),
            'ko', label='Crossing Points', color='red')
    ax[0].set_xlabel("Position (µsteps)")
    ax[0].set_xlim(0, 0.025e7)
    ax[0].set_ylim(-500, 500)
    ax[0].set_ylabel("Signal")
    ax[0].set_title('Position against Signal (zoomed in)')
    ax[0].legend(loc='upper right')
    ax[1].set_title('Histogram of the Crossing Point Distances')
    ax[1].hist(diff, bins=100, color='blue')
    ax[1].set_xlabel("Distance between crossings (µsteps)")
    ax[1].set_ylabel("Number of entries")

    # print("The mean difference between crossing points is %.0d and the standard\
    #     deviation between crossing points is %.0d" % (np.mean(diff),
    #     np.std(diff)))

    fig.tight_layout()
    fig.show()
    plt.savefig("figures/crossing_points.png")


    # step 5 - find the global calibration
    mean_diff = np.mean(diff) # difference in musteps between crossing points
    wl_ref = 532.0e-9  # from laser specifications
    # there are 2 crossing points in each wavelength
    metres_per_microstep = wl_ref / (2.0*mean_diff)

    # print('Distance moved per mu step is %.5e metres' % (metres_per_microstep))
    return metres_per_microstep
