import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys

# Builds a dictionary with references to .csv columns
def get_vals(file_path):
    try:
        data = pd.read_csv(file_path, skiprows=[0, 1])
    except FileNotFoundError:
        sys.exit(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        sys.exit(f"Empty data: {file_path}")
    except pd.errors.ParserError:
        sys.exit(f"Error parsing file: {file_path}")
    
    vals = {
        "matrix_size": data['matrix_size'],
        "mean_exec_time": data['mean_exec_time'],
        "stdev_exec_time": data['stdev_exec_time'],
        "mean_effective_bandwidth": data['mean_effective_bandwidth'],
        "stdev_effective_bandwidth": data['stdev_effective_bandwidth']
    }
    
    return vals

# Insert a smoothed interpolation of the given data
def add_line(x_vals, y_vals, y_stdev, line_color='blue', label=None):
    # Convert string labels to numerical values
    x_numeric = np.arange(len(x_vals))
    
    # Interpolate y-values and y_stdev
    interp_func = interp1d(x_numeric, y_vals, kind='quadratic')
    interp_stdev = interp1d(x_numeric, y_stdev, kind='quadratic')
    
    # Generate a finer grid for x-values
    x_fine = np.linspace(0, len(x_vals) - 1, 1000)
    
    # Evaluate interpolated functions at the finer grid
    y_smooth = interp_func(x_fine)
    y_stdev_smooth = interp_stdev(x_fine)
    
    # Plot the smooth line
    plt.plot(x_fine, y_smooth, color=line_color, label=label)
    # Fill between the lines
    plt.fill_between(x_fine, y_smooth - y_stdev_smooth, y_smooth + y_stdev_smooth, alpha=0.1, color=line_color)
    # Set x-axis tick labels
    plt.xticks(x_numeric, x_vals)

if __name__ == "__main__":
    
    ##
    # USER DEFINED DATA
    #   add as many blocks as lines you need to plot
    #   (respecting the provided format)
    ##
    inputs = [
        {
            "file_name": "data/blocks-gpu-coalesced_6-to-15-steps_5-blocksize_3-thX_5-thY.csv",
            "line_color": "red",
            "label": "blocks_coalesced-GPU"
        },
        {
            "file_name": "data/blocks-gpu_6-to-15-steps_5-blocksize_3-thX_5-thY.csv",
            "line_color": "green",
            "label": "blocks-GPU"
        },
        {
            "file_name": "data/data-blocks_version_O2.csv",
            "line_color": "blue",
            "label": "blocks-CPU (-O2)"
        }
    ]

    # Begin processing
    data = [get_vals(obj["file_name"]) for obj in inputs]
    
    # Display (matrix_size X mean_exec_time) graph
    plt.figure(figsize=(10, 5))
    for key, elem in enumerate(data):
        obj = inputs[key]
        add_line(elem["matrix_size"], elem["mean_exec_time"], elem["stdev_exec_time"], line_color=obj["line_color"], label=obj["label"])

    # Customize the graph
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.legend(loc='upper left')
    plt.xlim(0, len(data[0]["mean_exec_time"]) - 1)
    plt.grid(axis='y')
    plt.show()

    # Display (matrix_size X mean_effective_bandwidth) graph
    plt.figure(figsize=(10, 5))
    for key, elem in enumerate(data):
        obj = inputs[key]
        add_line(elem["matrix_size"], elem["mean_effective_bandwidth"], elem["stdev_effective_bandwidth"], line_color=obj["line_color"], label=obj["label"])

    # Customize the graph
    plt.xlabel('Matrix Size')
    plt.ylabel('Effective Bandwidth (GB/s)')
    plt.legend(loc='upper left')
    plt.xlim(0, len(data[0]["mean_effective_bandwidth"]) - 1)
    plt.grid(axis='y')
    plt.show()
