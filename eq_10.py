from workflow.internal import load_dataset, normalization, print_matrix
from workflow.clustering_functions import labels_to_partition, partition_to_labels, labels_to_partition
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from networkx.algorithms import community
from workflow.spectral_clustering import *


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    Y, name, dimension, account_prop, trans_matrix = load_dataset()

    # Extract time series
    # extract_timeseries(Y, name)

    # Normalise time series
    Y_norm = normalization(Y, name)

    # Compute kNN
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(Y_norm)
    distances, indices = nbrs.kneighbors(Y_norm)

    # Estimate sigma as median distance if not set
    sigma = np.median(distances[:, 1:])  # skip self-distance (0)

    rows, cols, weights = [], [], []

    for i in range(Y_norm.shape[0]):
        for j in range(1, k + 1):  # skip self-distance
            neighbor = indices[i, j]
            dist = distances[i, j]
            sim = np.exp(-dist**2 / (2 * sigma**2)) # Computes similarity weight using Gaussian kernel: exp(-distance²/(2σ²))
            # The Gaussian kernel calculates a similarity score between two input vectors, based on the distance between them in a high-dimensional feature space

            # Adds directed edge (i -> neighbor) with the weight
            rows.append(i)
            cols.append(neighbor)
            weights.append(sim)

            # Adds reverse edge (neighbor -> i) to make the graph symmetric
            rows.append(neighbor)
            cols.append(i)
            weights.append(sim)

    # Create sparse similarity matrix
    graph = csr_matrix((weights, (rows, cols)), shape=(Y_norm.shape[0], Y_norm.shape[0]))

    print_matrix(graph)

    # l_norm = compute_normalized_laplacian(graph)

    # eigenvalues = compute_eigenvalues(l_norm)

    # relative_eigengaps, k_values = compute_relative_eigengap(eigenvalues)

    optimal_k, eigenvectors = find_optimal_clusters(graph)

    labels = compute_spectral_clustering(eigenvectors, optimal_k)

    print(labels)

    partition = labels_to_partition(labels)
    print(partition)
