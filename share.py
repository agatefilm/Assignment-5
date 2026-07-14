'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.

'''
#%%
import matplotlib
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()


def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()


def get_ground_level(pcd):
    """
    Find the best value for the ground level using histogram.
    The ground level is the most common Z value (mode).
    
    Args:
        pcd: numpy array of shape (n_points, 3) with X, Y, Z coordinates
    
    Returns:
        float: ground level (Z coordinate)
    """
    z_values = pcd[:, 2]
    
    # Create histogram to find the most common Z value
    # Use adaptive binning based on Freedman-Diaconis rule
    q1, q3 = np.percentile(z_values, [25, 75])
    iqr = q3 - q1
    bin_width = 2 * iqr / (len(z_values) ** (1/3))
    n_bins = int((z_values.max() - z_values.min()) / bin_width)
    n_bins = max(50, min(n_bins, 200))  # Limit bins between 50 and 200
    
    hist, bin_edges = np.histogram(z_values, bins=n_bins)
    
    # Find the bin with maximum count
    max_count_idx = np.argmax(hist)
    ground_level = (bin_edges[max_count_idx] + bin_edges[max_count_idx + 1]) / 2
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(z_values, bins=n_bins, color='blue', alpha=0.7)
    
    # Highlight the ground level bin
    ground_bin_idx = np.digitize(ground_level, bins) - 1
    if 0 <= ground_bin_idx < len(patches):
        patches[ground_bin_idx].set_facecolor('red')
        patches[ground_bin_idx].set_alpha(0.9)
    
    plt.axvline(x=ground_level, color='red', linestyle='-', 
                label=f'Ground Level: {ground_level:.2f} m')
    plt.title('Z-coordinate Histogram')
    plt.xlabel('Z coordinate (m)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return ground_level
