from workflow.internal import load_dataset, knn_graph, normalization, visualize_metrics, extract_timeseries
from workflow.SQUIC_functions import squic_fit_matrix_computation
from workflow.clustering_functions import clustering_2_communities, modularity_fscore, plot_Q_f1, labels_to_partition, partition_to_labels
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

from networkx.algorithms import community


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    Y, name, dimension, account_prop, trans_matrix = load_dataset()

    # Extract time series
    extract_timeseries(Y, name)

    # Normalise time series
    Y_norm = normalization(Y, name)

    n_neighbors = 20
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', n_jobs=-1)
    nbrs.fit(Y_norm)
    knn_matrix = nbrs.kneighbors_graph(Y_norm, mode='connectivity')

    G_knn = nx.from_scipy_sparse_array(knn_matrix)

    print("++++++++++++++++++++++++++++++++++++++++++++++")
    if nx.is_connected(G_knn):
        print(f"Graph given by KNN on timeseries is connected")
    else:
        components = list(nx.connected_components(G_knn))
        main_component = max(components, key=len)
        isolated_nodes = [list(comp)[0] for comp in components if len(comp) == 1]
        
        print(f"Main component: {len(main_component)} nodes")
        print(f"Isolated nodes: {len(isolated_nodes)} nodes")
        print(f"Other components: {len(components) - len(isolated_nodes) - 1}\n")
    print("++++++++++++++++++++++++++++++++++++++++++++++")

    # Print knn matrix
    plt.figure(figsize=(7, 7))
    plt.spy(knn_matrix, markersize=5)
    plt.ylabel("Users", fontsize=18)

    plt.tick_params(axis='x')
    plt.tick_params(axis='y') 
    plt.show()

    results_squic = {}

    # Run SQUIC_fit
    results_squic['squic-fit-matrix'] = squic_fit_matrix_computation(Y_norm, name, dimension, knn_matrix, printMatrix=True)

    dict_cluster = clustering_2_communities(results_squic, method='implemented')

    metrics = modularity_fscore(dict_cluster, results_squic, account_prop)

    for _, data in metrics.items():
        for rho, results in data.items():
            print(f"for rho = {rho} : {results}")

    plot_Q_f1(metrics)
    