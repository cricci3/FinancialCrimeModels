import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import networkx as nx
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


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


def compute_eigenvalues(laplacian_matrix):
    """
    Compute the smallest k eigenvalues of the Laplacian matrix (sparse).
    """
    n = laplacian_matrix.shape[0]
    k = n - 1  # eigsh cannot compute more than n-1 eigenvalues

    # If small enough, dense fallback
    # if n <= 500:  # you can adjust this threshold
    #     dense_L = laplacian_matrix.toarray()
    #     eigenvalues = eigh(dense_L, eigvals_only=True)
    # else:
    #     # Sparse: compute k smallest eigenvalues
    #     try:
    #         eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=k, which='SM')
    #         eigenvalues = np.sort(eigenvalues)
    #     except Exception as e:
    #         raise RuntimeError(f"Sparse eigenvalue computation failed: {e}")

    eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=k, which='SM')
    eigenvalues = np.sort(eigenvalues)

    return eigenvalues, eigenvectors


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
    # Convert NetworkX graph to sparse adjacency matrix if needed
    if isinstance(graph, nx.Graph):
        adjacency_matrix = nx.to_scipy_sparse_array(graph, format='csr')
    else:
        adjacency_matrix = graph  # assume already sparse

    # Compute normalized Laplacian
    L_norm = compute_normalized_laplacian(adjacency_matrix)

    # Compute eigenvalues
    eigenvalues, eigenvectors = compute_eigenvalues(L_norm)

    # Compute relative eigengaps
    relative_eigengaps, k_values = compute_relative_eigengap(eigenvalues)

    # Find optimal k (highest relative eigengap)
    if len(relative_eigengaps) > 0:
        print(f"len(relative_eigengaps) > 0 : True")
        optimal_k = k_values[np.argmax(relative_eigengaps)]
    else:
        optimal_k = 2  # default fallback

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

    return optimal_k, eigenvectors


def compute_spectral_clustering(eigenvectors, optimal_k, method='kmeans'):
    selected_eigenvectors = eigenvectors[:, :optimal_k]
    X_normalized = normalize(selected_eigenvectors, norm='l2')
    
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels = kmeans.fit_predict(X_normalized)

    elif method == 'hierarchical':
        # Use Ward’s method, single-link, complete-link, etc.
        Z = linkage(X_normalized, method='ward')

        # Plot dendrogram (optional)
        plt.figure(figsize=(10, 5))
        dendrogram(Z)
        plt.title("Dendrogram of Spectral Embedding")
        plt.show()

        # Choose a cut level, e.g., fcluster with t=optimal_k clusters:
        labels = fcluster(Z, t=optimal_k, criterion='maxclust')

    return labels
