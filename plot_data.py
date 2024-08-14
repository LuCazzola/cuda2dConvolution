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
        "kernel_size": data['kernel_size'],
        "matrix_size": data['matrix_size'],
        "mean_exec_time": data['mean_exec_time'],
        "stdev_exec_time": data['stdev_exec_time'],
        "mean_effective_bandwidth": data['mean_effective_bandwidth'],
        "stdev_effective_bandwidth": data['stdev_effective_bandwidth'],
        "mean_flops" : data['mean_flops'],
        "stdev_flops" : data['stdev_flops']
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
    plt.xticks(x_numeric, ['$' + val + '$' for val in x_vals])



if __name__ == "__main__":
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    ##
    # USER DEFINED DATA
    #   add as many blocks as lines you need to plot
    #   (respecting the provided format)
    ##
    inputs = [
        {
            "file_name": "data/CPU-conv-naive_3-to-9-kernel_6-to-11-size_5-iter.csv",
            "line_color": "red",
            "label": "Alg. 1"
        },
        {
            "file_name": "data/GPU-conv-naive_3-to-9-kernel_8-to-14-size_16-by-16-th-per-block_50-iter.csv",
            "line_color": "blue",
            "label": "Alg. 2"
        },
        {
            "file_name": "data/GPU-conv-shared_3-to-9-kernel_8-to-14-size_16-by-16-th-per-block_50-iter.csv",
            "line_color": "green",
            "label": "Alg. 3"
        },
        {
            "file_name": "data/GPU-conv-shared-constk_3-to-9-kernel_8-to-14-size_16-by-16-th-per-block_50-iter.csv",
            "line_color": "orange",
            "label": "Alg. 4"
        },
        {
            "file_name": "data/GPU-conv-shared-constk-cached_3-to-9-kernel_8-to-14-size_16-by-16-th-per-block_50-iter.csv",
            "line_color": "purple",
            "label": "Alg. 5"
        }
    ]

    # Begin processing
    data = [get_vals(obj["file_name"]) for obj in inputs]    
    unique_kernels = sorted(set(data[0]["kernel_size"]))

    '''
    For each algorithm compare all kernel sizes (matrix_size X mean_flops) + (matrix_size X mean_effective_bandwidth)
    '''

    # Display (matrix_size X mean_effective_bandwidth) graph
    for key, elem in enumerate(data):
        obj = inputs[key]

        plt.figure(figsize=(10, 5))
        for i, k_size in enumerate(unique_kernels) :
            # Filter data for the current kernel size
            idxs = [i for i, x in enumerate(elem['kernel_size']) if x == k_size]
            mat_sizes = [elem['matrix_size'][i] for i in idxs]
            mean_flops = [elem['mean_flops'][i] for i in idxs]
            stdev = [elem['stdev_flops'][i] for i in idxs]
            add_line(mat_sizes, mean_flops, stdev, line_color=colors[i], label=f"{obj['label']} [{k_size}]")

        # Customize the graph
        plt.xlabel('Matrix Size')
        plt.ylabel('TFLOPS')
        plt.legend(loc='upper left')
        plt.xlim(0, len(mean_flops) - 1)
        plt.grid(axis='y')
        plt.show()

        plt.figure(figsize=(10, 5))
        for i, k_size in enumerate(unique_kernels) :
            # Filter data for the current kernel size
            idxs = [i for i, x in enumerate(elem['kernel_size']) if x == k_size]
            mat_sizes = [elem['matrix_size'][i] for i in idxs]
            mean_bandwidth = [elem['mean_effective_bandwidth'][i] for i in idxs]
            stdev = [elem['stdev_effective_bandwidth'][i] for i in idxs]
            add_line(mat_sizes, mean_bandwidth, stdev, line_color=colors[i], label=f"{obj['label']} [{k_size}]")

        # Customize the graph
        plt.xlabel('Matrix Size')
        plt.ylabel('Effective Bandwidth (GB/s)')
        plt.legend(loc='upper left')
        plt.xlim(0, len(mean_bandwidth) - 1)
        plt.grid(axis='y')
        plt.show()

    
    """
    For each Kernel size compare all algorithms (matrix_size X mean_flops) + (matrix_size X mean_effective_bandwidth)
    """ 

    # Display graphs per matrix size for fixed kernel sizes
    for kernel_size in unique_kernels:
        plt.figure(figsize=(10, 5))
        for key, elem in enumerate(data):
            obj = inputs[key]
            # Filter data for the current kernel size
            idxs = [i for i, x in enumerate(elem['kernel_size']) if x == kernel_size]
            if idxs:
                mat_sizes = [elem['matrix_size'][i] for i in idxs]
                mean_bandwidth = [elem['mean_effective_bandwidth'][i] for i in idxs]
                stdev_bandwidth = [elem['stdev_effective_bandwidth'][i] for i in idxs]
                add_line(mat_sizes, mean_bandwidth, stdev_bandwidth, line_color=obj["line_color"], label=obj["label"])
        
        # Customize the graph
        plt.xlabel('Matrix Size')
        plt.ylabel('Effective Bandwidth (GB/s)')
        plt.legend(loc='upper left')
        plt.title(f'Kernel Size: {kernel_size}')
        plt.grid(axis='y')
        plt.xlim(0, len(mean_bandwidth) - 1)
        plt.show()

        plt.figure(figsize=(10, 5))
        for key, elem in enumerate(data):
            obj = inputs[key]
            # Filter data for the current kernel size
            idxs = [i for i, x in enumerate(elem['kernel_size']) if x == kernel_size]
            if idxs:
                mat_sizes = [elem['matrix_size'][i] for i in idxs]
                mean_flops = [elem['mean_flops'][i] for i in idxs]
                stdev_flops = [elem['stdev_flops'][i] for i in idxs]
                add_line(mat_sizes, mean_flops, stdev_flops, line_color=obj["line_color"], label=obj["label"])
        
        # Customize the graph
        plt.xlabel('Matrix Size')
        plt.ylabel('TFLOPS')
        plt.legend(loc='upper left')
        plt.title(f'Kernel Size: {kernel_size}')
        plt.grid(axis='y')
        plt.xlim(0, len(mean_flops) - 1)
        plt.show()