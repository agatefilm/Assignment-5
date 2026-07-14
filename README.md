# Assignment 5 - Point Cloud Data Processing

## Task 1: Ground Level Detection

**Method:** Histogram analysis with adaptive binning (Freedman-Diaconis rule)

**Results:**
- Dataset 1: Ground level = **61.10 m**
- Dataset 2: Ground level = **61.13 m**

**Plots:**

### Dataset 1 - Ground Level Histogram
![Dataset 1 Ground Level](plots/dataset1_ground_level_histogram.png)

### Dataset 2 - Ground Level Histogram  
![Dataset 2 Ground Level](plots/dataset2_ground_level_histogram.png)

---

## Task 2: Optimal Epsilon for DBSCAN

**Method:** k-distance graph (k=4) with elbow method, 98th percentile

**Results:**
- Dataset 1: Optimal epsilon = **0.72**
- Dataset 2: Optimal epsilon = **0.63**

**Plots:**

### Dataset 1 - Elbow Plot
![Dataset 1 Elbow](plots/dataset1_elbow_plot.png)

### Dataset 2 - Elbow Plot
![Dataset 2 Elbow](plots/dataset2_elbow_plot.png)

### DBSCAN Clustering

#### Dataset 1 - 2D Clusters
![Dataset 1 DBSCAN 2D](plots/dataset1_dbscan_clustering.png)

#### Dataset 1 - 3D Clusters
![Dataset 1 DBSCAN 3D](plots/dataset1_dbscan_clustering_3d.png)

#### Dataset 2 - 2D Clusters
![Dataset 2 DBSCAN 2D](plots/dataset2_dbscan_clustering.png)

#### Dataset 2 - 3D Clusters
![Dataset 2 DBSCAN 3D](plots/dataset2_dbscan_clustering_3d.png)

---

## Task 3: Catenary Identification

**Method:** Largest cluster by X-Y area span

**Results:**

### Dataset 1
- Catenary cluster: Cluster 0
- Area = **3278.55 m²**
- min(x) = **21.15 m**
- min(y) = **80.01 m**
- max(x) = **62.14 m**
- max(y) = **160.00 m**

### Dataset 2
- Catenary cluster: Cluster 0
- Area = **2692.76 m²**
- min(x) = **7.39 m**
- min(y) = **0.00 m**
- max(x) = **41.05 m**
- max(y) = **80.00 m**

**Plots:**

### Dataset 1 - Catenary (2D with bounding box)
![Dataset 1 Catenary](plots/dataset1_catenary.png)

### Dataset 1 - Catenary (3D)
![Dataset 1 Catenary 3D](plots/dataset1_catenary_3d.png)

### Dataset 2 - Catenary (2D with bounding box)
![Dataset 2 Catenary](plots/dataset2_catenary.png)

### Dataset 2 - Catenary (3D)
![Dataset 2 Catenary 3D](plots/dataset2_catenary_3d.png)

---

## Summary

**Dataset 1:**
- Ground level = 61.10 m
- Optimal epsilon = 0.72
- Catenary area = 3278.55 m², bounds: x=[21.15, 62.14], y=[80.01, 160.00]

**Dataset 2:**
- Ground level = 61.13 m
- Optimal epsilon = 0.63
- Catenary area = 2692.76 m², bounds: x=[7.39, 41.05], y=[0.00, 80.00]

---

## Submission

**Highest task attempted: Task 3**

**Dataset 2 results:**
- Ground level = 61.13 m
- Optimal epsilon = 0.63
- Area of the catenary cluster = 2692.76 m²
