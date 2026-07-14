"""
Assignment 5 - Point Cloud Data Processing for Structural Health Monitoring

NOTE: The ground is now filtered out before clustering to ensure the catenary
cluster does not include the ground surface, which was causing the reported area
to be too large.

Complete solution for all three tasks with ground filtering.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create output directory for plots
os.makedirs('plots', exist_ok=True)


def get_ground_level(pcd, save_path=None):
    """
    Find the best value for the ground level using histogram analysis.
    
    Args:
        pcd: numpy array of shape (n_points, 3) with X, Y, Z coordinates
        save_path: optional path to save the histogram plot
    
    Returns:
        float: ground level (Z coordinate in meters)
    """
    z_values = pcd[:, 2]
    
    # Use Freedman-Diaconis rule for optimal bin size
    q1, q3 = np.percentile(z_values, [25, 75])
    iqr = q3 - q1
    bin_width = 2 * iqr / (len(z_values) ** (1/3))
    n_bins = int((z_values.max() - z_values.min()) / bin_width)
    n_bins = max(50, min(n_bins, 200))
    
    hist, bin_edges = np.histogram(z_values, bins=n_bins)
    max_count_idx = np.argmax(hist)
    ground_level = (bin_edges[max_count_idx] + bin_edges[max_count_idx + 1]) / 2
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(z_values, bins=n_bins, color='blue', alpha=0.7)
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return ground_level


def find_optimal_epsilon(pcd, k=5, percentile=98, save_path=None):
    """
    Find optimal epsilon value for DBSCAN using the k-distance graph.
    Uses percentile-based selection as a proxy for the elbow point.
    
    Args:
        pcd: numpy array of shape (n_points, 3) with X, Y, Z coordinates
        k: number of nearest neighbors (default 5)
        percentile: percentile to use as optimal epsilon (default 98)
        save_path: optional path to save the elbow plot
    
    Returns:
        float: optimal epsilon value
    """
    xy_data = pcd[:, :2]
    nbrs = NearestNeighbors(n_neighbors=k).fit(xy_data)
    distances, _ = nbrs.kneighbors(xy_data)
    k_distances = distances[:, -1]
    k_distances_sorted = np.sort(k_distances)
    
    percentile_idx = int(len(k_distances_sorted) * percentile / 100)
    optimal_epsilon = k_distances_sorted[percentile_idx]
    
    # Plot k-distance graph
    plt.figure(figsize=(12, 8))
    x = range(1, len(k_distances_sorted) + 1)
    plt.plot(x, k_distances_sorted, 'b-', linewidth=1)
    plt.axvline(x=percentile_idx, color='red', linestyle='--', 
                label=f'{percentile}th percentile: ε = {optimal_epsilon:.4f}')
    # Note: Using percentile as proxy for elbow point
    plt.title(f'{k}-Distance Graph (Percentile-based Epsilon Selection)')
    plt.xlabel('Points sorted by distance to their k-th nearest neighbor')
    plt.ylabel(f'{k}-NN distance (m)')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return optimal_epsilon


def apply_dbscan(pcd, epsilon, min_samples=5, save_path=None):
    """
    Apply DBSCAN clustering.
    
    Args:
        pcd: numpy array of shape (n_points, 3) with X, Y, Z coordinates
        epsilon: maximum distance for DBSCAN
        min_samples: minimum samples for DBSCAN (default 5)
        save_path: optional path to save the clustering plot
    
    Returns:
        labels: cluster labels for each point
    """
    xy_data = pcd[:, :2]
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(xy_data)
    return labels


def visualize_dbscan(pcd, labels, epsilon, save_path=None):
    """
    Visualize DBSCAN clustering results.
    
    Args:
        pcd: numpy array of shape (n_points, 3) with X, Y, Z coordinates
        labels: cluster labels for each point
        epsilon: epsilon value used for DBSCAN
        save_path: optional path to save the plot
    """
    unique_labels = set(labels)
    n_colors = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
    
    # 2D plot
    plt.figure(figsize=(12, 8))
    for j, label in enumerate(sorted(unique_labels)):
        if label == -1:
            col = [0, 0, 0, 0.5]
        else:
            col = colors[j % len(colors)]
        cluster_points = pcd[labels == label]
        s = 0.5 if label == -1 else (2 if np.sum(labels == label) > 10000 else 1)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[col], s=s, alpha=0.7)
    plt.title(f'DBSCAN Clustering (ε={epsilon:.2f}, min_samples=5)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for j, label in enumerate(sorted(unique_labels)):
        if label == -1:
            col = [0, 0, 0, 0.3]
        else:
            col = colors[j % len(colors)]
        cluster_points = pcd[labels == label]
        s = 0.5 if label == -1 else (1 if np.sum(labels == label) > 10000 else 0.5)
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                   c=[col], s=s, alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D DBSCAN Clustering (ε={epsilon:.2f}, min_samples=5)')
    if save_path:
        plt.savefig(save_path.replace('.png', '_3d.png'), dpi=150, bbox_inches='tight')
    plt.close()


def find_largest_cluster(pcd, labels):
    """
    Find the largest cluster by area (x,y span).
    
    Args:
        pcd: numpy array of shape (n_points, 3) with X, Y, Z coordinates
        labels: cluster labels for each point
    
    Returns:
        tuple: (largest_cluster_label, area, min_x, min_y, max_x, max_y)
    """
    unique_labels = set(labels)
    cluster_info = []
    
    for label in sorted(unique_labels):
        if label == -1:
            continue
        cluster_points = pcd[labels == label]
        if len(cluster_points) == 0:
            continue
        min_x, max_x = cluster_points[:, 0].min(), cluster_points[:, 0].max()
        min_y, max_y = cluster_points[:, 1].min(), cluster_points[:, 1].max()
        area = (max_x - min_x) * (max_y - min_y)
        
        # Calculate linearity using PCA
        xy_points = cluster_points[:, :2]
        if len(xy_points) > 1:
            pca = PCA(n_components=2)
            pca.fit(xy_points)
            # Linearity: ratio of first to second eigenvalue (higher = more linear)
            linearity = pca.explained_variance_[0] / pca.explained_variance_[1] if pca.explained_variance_[1] > 0 else 0
        else:
            linearity = 0
        
        cluster_info.append((label, area, len(cluster_points), linearity, min_x, min_y, max_x, max_y))
    
    # Sort by area (descending)
    cluster_info.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 5 clusters by area (excluding noise)
    print("\nTop 5 clusters by area:")
    for label, area, n_points, linearity, min_x, min_y, max_x, max_y in cluster_info[:5]:
        print(f"  Cluster {label}: Area={area:.2f} m², Points={n_points}, Linearity={linearity:.2f}")
    
    if cluster_info:
        largest_label = cluster_info[0][0]
        largest_area = cluster_info[0][1]
        min_x = cluster_info[0][4]
        min_y = cluster_info[0][5]
        max_x = cluster_info[0][6]
        max_y = cluster_info[0][7]
        return largest_label, largest_area, min_x, min_y, max_x, max_y
    return -1, 0, 0, 0, 0, 0


def visualize_catenary(pcd, labels, largest_label, save_path=None):
    """
    Visualize the catenary cluster.
    
    Args:
        pcd: numpy array of shape (n_points, 3) with X, Y, Z coordinates
        labels: cluster labels for each point
        largest_label: label of the largest cluster (catenary)
        save_path: optional path to save the plot
    """
    if largest_label == -1:
        return
    
    largest_points = pcd[labels == largest_label]
    other_points = pcd[labels != largest_label]
    
    min_x, max_x = largest_points[:, 0].min(), largest_points[:, 0].max()
    min_y, max_y = largest_points[:, 1].min(), largest_points[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)
    
    # 2D plot
    plt.figure(figsize=(12, 8))
    plt.scatter(other_points[:, 0], other_points[:, 1], c='gray', s=0.5, alpha=0.3, label='Other points')
    plt.scatter(largest_points[:, 0], largest_points[:, 1], c='red', s=2, label='Catenary', alpha=0.8)
    rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                       fill=False, edgecolor='blue', linewidth=2, linestyle='--')
    plt.gca().add_patch(rect)
    plt.title(f'Catenary Cluster (Area: {area:.2f} m²)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_points[:, 0], other_points[:, 1], other_points[:, 2], 
               c='gray', s=0.5, alpha=0.2, label='Other points')
    ax.scatter(largest_points[:, 0], largest_points[:, 1], largest_points[:, 2], 
               c='red', s=2, label='Catenary', alpha=0.8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Catenary Cluster (Area: {area:.2f} m²)')
    ax.legend()
    if save_path:
        plt.savefig(save_path.replace('.png', '_3d.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Assignment 5 - Point Cloud Data Processing")
    print("NOTE: Ground is filtered out before clustering")
    print("="*70)
    
    # Process Dataset 1
    print("\n" + "="*70)
    print("Processing Dataset 1...")
    print("="*70)
    pcd1 = np.load("dataset1.npy")
    
    # Task 1: Find ground level
    print("\nTask 1: Finding ground level...")
    gl1 = get_ground_level(pcd1, save_path='plots/dataset1_ground_level_histogram.png')
    print(f"Ground level: {gl1:.4f} m")
    
    # Filter out ground
    pcd1_above = pcd1[pcd1[:, 2] > gl1]
    print(f"Points above ground: {len(pcd1_above)} (original: {len(pcd1)})")
    
    # Task 2: Find optimal epsilon (using k=5, min_samples=5)
    print("\nTask 2: Finding optimal epsilon...")
    eps1 = find_optimal_epsilon(pcd1_above, k=5, save_path='plots/dataset1_elbow_plot.png')
    print(f"Optimal epsilon: {eps1:.4f} m")
    
    # Apply DBSCAN with filtered data
    labels1 = apply_dbscan(pcd1_above, epsilon=eps1, min_samples=5)
    visualize_dbscan(pcd1_above, labels1, eps1, save_path='plots/dataset1_dbscan_clustering.png')
    
    # Task 3: Find catenary
    print("\nTask 3: Finding catenary...")
    cat_label1, cat_area1, min_x1, min_y1, max_x1, max_y1 = find_largest_cluster(pcd1_above, labels1)
    print(f"Catenary: Cluster {cat_label1}, Area: {cat_area1:.4f} m²")
    print(f"Bounds: x=[{min_x1:.4f}, {max_x1:.4f}], y=[{min_y1:.4f}, {max_y1:.4f}]")
    visualize_catenary(pcd1_above, labels1, cat_label1, save_path='plots/dataset1_catenary.png')
    
    # Process Dataset 2
    print("\n" + "="*70)
    print("Processing Dataset 2...")
    print("="*70)
    pcd2 = np.load("dataset2.npy")
    
    # Task 1: Find ground level
    print("\nTask 1: Finding ground level...")
    gl2 = get_ground_level(pcd2, save_path='plots/dataset2_ground_level_histogram.png')
    print(f"Ground level: {gl2:.4f} m")
    
    # Filter out ground
    pcd2_above = pcd2[pcd2[:, 2] > gl2]
    print(f"Points above ground: {len(pcd2_above)} (original: {len(pcd2)})")
    
    # Task 2: Find optimal epsilon (using k=5, min_samples=5)
    print("\nTask 2: Finding optimal epsilon...")
    eps2 = find_optimal_epsilon(pcd2_above, k=5, save_path='plots/dataset2_elbow_plot.png')
    print(f"Optimal epsilon: {eps2:.4f} m")
    
    # Apply DBSCAN with filtered data
    labels2 = apply_dbscan(pcd2_above, epsilon=eps2, min_samples=5)
    visualize_dbscan(pcd2_above, labels2, eps2, save_path='plots/dataset2_dbscan_clustering.png')
    
    # Task 3: Find catenary
    print("\nTask 3: Finding catenary...")
    cat_label2, cat_area2, min_x2, min_y2, max_x2, max_y2 = find_largest_cluster(pcd2_above, labels2)
    print(f"Catenary: Cluster {cat_label2}, Area: {cat_area2:.4f} m²")
    print(f"Bounds: x=[{min_x2:.4f}, {max_x2:.4f}], y=[{min_y2:.4f}, {max_y2:.4f}]")
    visualize_catenary(pcd2_above, labels2, cat_label2, save_path='plots/dataset2_catenary.png')
    
    # Save results
    with open('results.txt', 'w') as f:
        f.write("Assignment 5 Results\n")
        f.write("="*70 + "\n\n")
        f.write("Dataset 1:\n")
        f.write(f"  Ground level = {gl1:.2f} m\n")
        f.write(f"  Optimal epsilon = {eps1:.2f}\n")
        f.write(f"  Area of the catenary cluster = {cat_area1:.2f} m²\n")
        f.write(f"  Catenary bounds: x=[{min_x1:.2f}, {max_x1:.2f}], y=[{min_y1:.2f}, {max_y1:.2f}]\n\n")
        f.write("Dataset 2:\n")
        f.write(f"  Ground level = {gl2:.2f} m\n")
        f.write(f"  Optimal epsilon = {eps2:.2f}\n")
        f.write(f"  Area of the catenary cluster = {cat_area2:.2f} m²\n")
        f.write(f"  Catenary bounds: x=[{min_x2:.2f}, {max_x2:.2f}], y=[{min_y2:.2f}, {max_y2:.2f}]\n")
    
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print("="*70)
    print(f"Dataset 1:")
    print(f"  Ground level = {gl1:.2f} m")
    print(f"  Optimal epsilon = {eps1:.2f}")
    print(f"  Area of the catenary cluster = {cat_area1:.2f} m²")
    print(f"  Catenary bounds: x=[{min_x1:.2f}, {max_x1:.2f}], y=[{min_y1:.2f}, {max_y1:.2f}]")
    
    print(f"\nDataset 2:")
    print(f"  Ground level = {gl2:.2f} m")
    print(f"  Optimal epsilon = {eps2:.2f}")
    print(f"  Area of the catenary cluster = {cat_area2:.2f} m²")
    print(f"  Catenary bounds: x=[{min_x2:.2f}, {max_x2:.2f}], y=[{min_y2:.2f}, {max_y2:.2f}]")
    
    print("\n" + "="*70)
    print("SUBMISSION COMMENT:")
    print("="*70)
    print("Highest task attempted: Task 3")
    print("Dataset 2 results:")
    print(f"  Ground level = {gl2:.2f} m")
    print(f"  Optimal epsilon = {eps2:.2f}")
    print(f"  Area of the catenary cluster = {cat_area2:.2f} m²")
