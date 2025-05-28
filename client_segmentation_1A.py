from workflow.internal import load_dataset, knn_graph, normalization, visualize_metrics
from workflow.SQUIC_functions import squic_fit_matrix_computation
from workflow.clustering_functions import clustering_2_communities, modularity_fscore, plot_Q_f1, labels_to_partition, partition_to_labels
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import LabelEncoder
from networkx.algorithms import community
from sklearn.metrics import f1_score


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    Y, name, dimension, account_prop, trans_matrix = load_dataset()

    # bias_graph = knn_graph(trans_matrix)

    # node_degrees = bias_graph.getnnz(axis=1)

    # Extract time series
    # extract_timeseries(Y, name)

    # Normalise time series
    Y_norm = normalization(Y, name)

    # n_neighbors = 10
    # nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', n_jobs=-1)
    # nbrs.fit(Y_norm)
    # knn_g = nbrs.kneighbors_graph(Y_norm, mode='connectivity')

    # results_squic = {}

    # # Run SQUIC_fit
    # results_squic['squic-fit-matrix'] = squic_fit_matrix_computation(Y_norm, name, dimension, knn_g, printMatrix=False)

    # Compute kNN (Euclidean by default)
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
            sim = np.exp(-dist**2 / (2 * sigma**2))

            rows.append(i)
            cols.append(neighbor)
            weights.append(sim)

            # Make symmetric (undirected graph)
            rows.append(neighbor)
            cols.append(i)
            weights.append(sim)

    # Create sparse similarity matrix
    graph = csr_matrix((weights, (rows, cols)), shape=(Y_norm.shape[0], Y_norm.shape[0]))

    #plt.figure(figsize=(10, 10), dpi=300)
    plt.figure(figsize=(7, 7))
    # plt.spy(X, markersize=5, c="#484154")
    plt.spy(graph, markersize=5)
    # plt.xlabel("Users", fontsize=18)
    plt.ylabel("Users", fontsize=18)
    # plt.ylabel("Users")

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)  
    # plt.title("Sparsity Pattern of Precision Matrix (X)")
    
    plt.show()

    # dict_cluster = clustering_2_communities(graph, name, account_prop)
    dict_cluster = {
        "squic-fit-matrix" : {}
    }
    # Ensure all off-diagonal entries are positive
    X = np.abs(graph) 

    # Create a graph from matrix X
    G = nx.from_scipy_sparse_array(X)
    print(f"graph is connected {nx.is_connected(G)}")

    clustering = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='cluster_qr')
    labels_spectral = clustering.fit_predict(X)

    # Convert labels to list of sets
    partition_spectral = labels_to_partition(labels_spectral)
    dict_cluster['squic-fit-matrix'] = partition_spectral

    metrics = {
        method: {
                'spectral': {}
        } for method in dict_cluster.keys()
    }

    n = len(account_prop)
    class_labels = np.array([account_prop[i]["class"] for i in range(n)])

    # Convert class labels to 0/1
    le = LabelEncoder()
    true_labels = le.fit_transform(class_labels)

    # Convert partition (list of sets) to labels
    partition = dict_cluster['squic-fit-matrix']
    cluster_labels = partition_to_labels(partition, n)

    # Compute modularity
    mod = community.modularity(G, partition)
    metrics['squic-fit-matrix']['spectral']['modularity'] = round(float(mod), 4)

    # Number of clusters
    metrics['squic-fit-matrix']['spectral']['nCluster'] = len(partition)

    # Compute F1-score (test both label alignments)
    f1a = f1_score(true_labels, 1 - cluster_labels, average='weighted')
    f1b = f1_score(true_labels, cluster_labels, average='weighted')

    f1 = max(f1a, f1b)

    metrics['squic-fit-matrix']['spectral']['f1'] = round(f1, 2)

    print()
    print(f"Metrics")
    print(f"Modularity : {round(float(mod), 4)}")
    print(f"F1 : {round(f1, 2)}")