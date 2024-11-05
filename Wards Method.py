import pandas as pd 
import numpy as np 

#   Very simple
#   ESS = Σ Σ (xij - x̄i)²
#   xiv is the j'th observation of i'th cluster, x̄i is the mean of i'th cluster

#   Wards calculates the increase in ESS that would result in merging two clusters
#   ΔE(p,q) = [ (npnq)/(np+nq) ] ||x̄p - x̄q||²
#   np and nq are the number of observations in cluster p and q (The sizes)
#   x̄p and x̄q are the centroids of p and q
#   ||x̄p - x̄q|| is the Euclidean distance between p and q

#   (|p| . |q| / |p u q|) . ||x̄p - x̄q||²

def euclidean_distance(centroid1, centroid2):
    return np.sqrt(np.sum((centroid1 - centroid2) ** 2))

def calculate_centroid(data, cluster_label):
    cluster_mask = cluster_labels == cluster_label
    cluster_points = data[cluster_mask]
    return np.mean(cluster_points, axis=0)

def merge_clusters(data, label1, label2):
    global cluster_labels
    cluster_labels[cluster_labels == label2] = label1
    new_distance_matrix, unique_labels = make_distance_matrix(data)
    return new_distance_matrix, unique_labels

def wards_criterion(data, label1, label2):
    cluster1_mask = cluster_labels == label1
    cluster2_mask = cluster_labels == label2
    n1 = np.sum(cluster1_mask)
    n2 = np.sum(cluster2_mask)
    
    centroid1 = calculate_centroid(data, label1)
    centroid2 = calculate_centroid(data, label2)
    squared_dist = euclidean_distance(centroid1, centroid2) ** 2
    ward_distance = (n1 * n2) / (n1 + n2) * squared_dist
    
    return ward_distance

def make_distance_matrix(data):
    ##Check your notebook for a more efficent data structure for this.
    ##Takes too long with a matrix
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    distance_matrix = np.zeros((n_clusters, n_clusters))
    
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)} #Index Mapping
    
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels[i+1:], i+1): #Only top half of the triangle
            distance = wards_criterion(data, label1, label2)
            idx1, idx2 = label_to_idx[label1], label_to_idx[label2]
            distance_matrix[idx1, idx2] = distance
            distance_matrix[idx2, idx1] = distance  #Matrix is symmetrical
    
    np.fill_diagonal(distance_matrix, np.inf) #So self clustering doesnt occur
    return distance_matrix, unique_labels

def hierarchical_cluster(data, n_clusters=2):
    global cluster_labels
    cluster_labels = np.arange(len(data))
    
    while len(np.unique(cluster_labels)) > n_clusters:
        distance_matrix, unique_labels = make_distance_matrix(data)
        min_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape) #For matrix
        label1, label2 = unique_labels[min_idx[0]], unique_labels[min_idx[1]]
        distance_matrix, unique_labels = merge_clusters(data, label1, label2)
        print(f"Merged {label1}, {label2}")
        print(unique_labels)
    
    return cluster_labels

df = pd.read_csv('ProcessedData.csv')
Y = df.to_numpy()
cluster_labels = np.arange(len(Y))

final_labels = hierarchical_cluster(Y, n_clusters=1)