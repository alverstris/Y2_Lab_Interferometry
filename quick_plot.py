###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import read_data_results3 as rd

#Step 1 get the data and the x position
file=r"Data\green_1_white_2_8.8to13.8.txt" #this is the data
results = rd.read_data3(file)

y1 = np.array(results[0])
y2 = np.array(results[1])

x=np.array(results[5])


def plot_two_datasets(x1, y1, x2, y2,
                      label1="Green Light",
                      label2="White Light",
                      xlabel="microsteps", ylabel="signal"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- First subplot ---
    axes[0].plot(x1, y1, marker='o', linestyle='-', label=label1)
    axes[0].set_title("Raw Interferograms Produced (Green)")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True)
    
    # --- Second subplot ---
    axes[1].plot(x2, y2, marker='o', linestyle='-', label=label2)
    axes[1].set_title("Raw Interferograms Produced (White)")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    
    axes[1].grid(True)
    
    
    plt.tight_layout()
    plt.show()

plot_two_datasets(x,y1, x, y2)
