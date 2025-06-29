import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import networkx as nx
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize
from scipy.sparse import issparse
import logging


def compute_normalized_laplacian(adj):
    """
    Compute the normalized Laplacian matrix L_norm = D^(-1/2) * L * D^(-1/2) (https://people.orie.cornell.edu/dpw/orie6334/Fall2016/lecture7.pdf)
    where L = D - A is the unnormalized Laplacian
    Keeps everything sparse.
    """
    # Compute degree vector
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # Avoid division by zero

    # D^(-1/2)
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))

    # Compute Laplacian L = D - A
    D = diags(degrees)
    L = D - adj

    # normalized Laplacian = D^(-1/2) * L * D^(-1/2)
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    return L_norm


def compute_eigenvalues_eigenvectors(L_norm, k=12, max_retries=3, print_info=False):
    """
    Computes the first `k` smallest eigenvalues and eigenvectors of a normalized Laplacian.
    Retries with adjusted parameters if convergence is not reached.
    """
    k = k+1
    for attempt in range(max_retries):
        try:
            eigenvalues, eigenvectors = eigsh(L_norm, k=k, which='SM', tol=0, maxiter=1000*(attempt+1))
            
            if len(eigenvalues) == k or (len(eigenvalues) > 2 and len(eigenvalues) < k):
                if print_info:
                    print(eigenvalues)
                # Sort
                idx = np.argsort(eigenvalues)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # Discard first eigen (~0)
                # eigenvalues[1:], eigenvectors[:, 1:]
                return eigenvalues, eigenvectors
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")

    print(f"Failed to compute {k} eigenvalues after {max_retries} retries.")
    return None, None


def compute_relative_eigengap(eigenvalues):
    """
    Compute the relative eigengap as defined in equation (10):
    delta_k = (lambda_{k+1} - lambda_k) / lambda_{k+1}
    """
    n = len(eigenvalues)
    relative_eigengaps = []
    k_values = []

    for k in range(2, n):
        lambda_k = eigenvalues[k - 1]
        lambda_k_plus_1 = eigenvalues[k]

        if lambda_k_plus_1 > 1e-10:
            delta_k = (lambda_k_plus_1 - lambda_k) / lambda_k_plus_1
            relative_eigengaps.append(delta_k)
            k_values.append(k)

    return np.array(relative_eigengaps), np.array(k_values)


def find_optimal_clusters(graph, plot=True):
    """
    Find the optimal number of clusters using relative eigengap analysis.
    Input graph: sparse adjacency matrix (csr_matrix) or NetworkX graph.
    """
    if isinstance(graph, nx.Graph):
        adjacency_matrix = nx.to_scipy_sparse_array(graph, format='csr')
    else:
        adjacency_matrix = graph 

    # Compute normalized Laplacian
    L_norm = compute_normalized_laplacian(adjacency_matrix)

    # Compute eigenvalues
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(L_norm)

    if eigenvalues is not None and eigenvectors is not None:
        # Compute relative eigengaps
        relative_eigengaps, k_values = compute_relative_eigengap(eigenvalues)

        # Find optimal k (highest relative eigengap)
        if len(relative_eigengaps) > 0:
            optimal_k = k_values[np.argmax(relative_eigengaps)]
        else:
            optimal_k = 2  # default

        if plot:
            plt.figure(figsize=(12, 5))
            # Plot eigenvalues
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
            plt.xlabel('Eigenvalue Index')
            plt.ylabel('Eigenvalue')
            plt.title('Laplacian Eigenvalues')
            plt.grid(True)

            # Plot relative eigengaps
            plt.subplot(1, 2, 2)
            plt.plot(k_values, relative_eigengaps, 'ro-')
            plt.axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal k = {optimal_k}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Relative Eigengap')
            plt.title('Relative Eigengaps (Equation 10)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()
    else:
        optimal_k = 2
        eigenvectors = None

    return optimal_k, eigenvectors


def compute_spectral_clustering(eigenvectors, k, method='kmeans', plot=False):
    selected_eigenvectors = eigenvectors[:, :k]
    X_normalized = normalize(selected_eigenvectors, norm='l2')
    
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_normalized)

    elif method == 'hierarchical':
        Z = linkage(X_normalized, method='ward')

        if plot == True:
            plt.figure(figsize=(10, 5))
            dendrogram(Z)
            plt.show()

        labels = fcluster(Z, t=k, criterion='maxclust')
    
    return labels
