import numpy as np

import networkx as nx
from networkx.algorithms import community
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, adjusted_rand_score

from tqdm import tqdm
import time
import os

from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict

from workflow.SQUIC_functions import print_matrix
import matplotlib.pyplot as plt
from workflow.spectral_clustering import find_optimal_clusters, compute_spectral_clustering, compute_normalized_laplacian, compute_eigenvalues_eigenvectors, debug_spectral_clustering, compute_normalized_laplacian_fixed, compute_eigenvalues_eigenvectors_fixed, diagnose_adjacency_matrix, remove_isolated_nodes
import igraph as ig
import leidenalg as la


def labels_to_partition(labels):
    """Convert label array [0,0,1,1,2,2] -> list of sets [{0,1},{2,3},{4,5}]"""
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = set()
        clusters[label].add(idx)

    # Remove noise points if using DBSCAN (-1 labels)
    if -1 in clusters:
        noise_nodes = clusters.pop(-1)
        for node in noise_nodes:
            clusters[f'noise_{node}'] = {node}  # unique labels for noise

    return list(clusters.values())


def partition_to_labels(partition, n_nodes):
    """Convert list-of-sets partition to flat label list."""
    labels = np.zeros(n_nodes, dtype=int)
    for i, community in enumerate(partition):
        for node in community:
            labels[node] = i
    return labels


def clustering_same_n(W_matrices, name, account_prop):
    dict_cluster = {
        "louvain" : {},
        "spectral" : {},
        "dbscan" : {}
    }

    for rho, X in W_matrices.items():
        # Ensure all off-diagonal entries are positive
        X = np.abs(X) 

        # Ensure diagonal entries are zero
        X.setdiag(0) # SQUIC_Fit ensure that, SQUIC not, so manually turn into 0

        # Create a graph from matrix X
        G = nx.from_scipy_sparse_array(X)

        print(f"Graph for rho {rho} is connected? {nx.is_connected(G)}")

        # Louvain
        start = time.time()
        partition_louvain = community.louvain_communities(G)
        end_louv = time.time() - start
        dict_cluster['louvain'][rho] = partition_louvain

        # DBSCAN with multiple params
        best_diff = float('inf')
        best_params = None

        eps_range = np.linspace(0.1, 2.0, 20)
        min_samples_range = range(3, 10)

        eps_range = np.linspace(0.1, 2.0, 20)
        min_samples_range = range(3, 10)

        print(f"Trying different params for DBSCAN for lambda = {rho}")
        start = time.time()
        for eps in tqdm(eps_range):
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                labels = dbscan.fit_predict(X)
                
                # Ignore if all points are noise (-1) or single cluster
                if len(set(labels)) <= 1 or (set(labels) == {-1}):
                    continue

                partition = labels_to_partition(labels)

                diff = abs(len(partition) - len(dict_cluster['louvain'][rho]))
                if diff < best_diff:
                    best_diff = diff
                    best_params = (eps, min_samples)

        if best_params is not None:
            dbscan = DBSCAN(eps=best_params[0], min_samples=best_params[1], metric='cosine')
        else:
            print("No suitable DBSCAN parameters found!")
            dbscan = DBSCAN()

        labels_dbscan = dbscan.fit_predict(X)
        end_db = time.time() - start

        # Convert labels to list of sets
        partition_dbscan = labels_to_partition(labels_dbscan)
        dict_cluster['dbscan'][rho] = partition_dbscan

        # Use n_cluster = len(partitionin louvain)
        start = time.time()
        clustering = SpectralClustering(n_clusters=len(dict_cluster['louvain'][rho]), affinity='precomputed', assign_labels='cluster_qr')
        labels_spectral = clustering.fit_predict(X)
        end_spec = time.time() - start

        # Convert labels to list of sets
        partition_spectral = labels_to_partition(labels_spectral)

        dict_cluster['spectral'][rho] = partition_spectral

        # print(f"Number of louvain cluster for rho {rho} is {len(partition_louvain)}")
        # print(f"Number of DBSCAN cluster for rho {rho} is {len(partition_dbscan)}")
        # print(f"Number of Spectral cluster for rho {rho} is {len(partition_spectral)}\n")

        # print(f"Time for Louvain for lambda = {rho} -> {end_louv}")
        # print(f"Time for DBSCAN for lambda = {rho} -> {end_db}")
        # print(f"Time for Spectral for lambda = {rho} -> {end_spec}\n")
    
    return dict_cluster

def clustering_optimal_number(results_squic, plot=False):
    dict_cluster = {
        "louvain" : {},
        "spectral" : {},
        "dbscan" : {},
        "leiden" : {}
    }

    for technique, matrices in results_squic.items():
        for rho, X in matrices.items():
            # Ensure all off-diagonal entries are positive
            X = np.abs(X) 

            # Ensure diagonal entries are zero
            X.setdiag(0) # SQUIC_Fit ensure that, SQUIC not, so manually turn into 0

            # Create a graph from matrix X
            G = nx.from_scipy_sparse_array(X)

            # Louvain
            start = time.time()
            partition_louvain = community.louvain_communities(G)
            end_louv = time.time() - start
            dict_cluster['louvain'][rho] = partition_louvain

            # Leiden
            start = time.time()
            G_igraph = ig.Graph.from_networkx(G)
            partition_leiden = la.find_partition(G_igraph, la.ModularityVertexPartition)
            end_leiden = time.time() - start
            dict_cluster['leiden'][rho] = partition_leiden

            # DBSCAN with multiple params
            best_Q = -1
            best_params = None

            eps_range = np.linspace(0.1, 2.0, 20)
            min_samples_range = range(3, 10)

            print(f"Trying different params for DBSCAN for lambda = {rho}")
            start = time.time()
            for eps in tqdm(eps_range):
                for min_samples in min_samples_range:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    labels = dbscan.fit_predict(X)
                    
                    # Ignore if all points are noise (-1) or single cluster
                    if len(set(labels)) <= 1 or (set(labels) == {-1}):
                        continue

                    partition_dbscan = labels_to_partition(labels)

                    # compute Q to evaluate
                    db_Q = community.modularity(G, partition_dbscan)
                    if db_Q > best_Q:
                        best_Q = db_Q
                        best_params = (eps, min_samples)

            if best_params is not None:
                dbscan = DBSCAN(eps=best_params[0], min_samples=best_params[1], metric='cosine')
            else:
                print("No suitable DBSCAN parameters found! DBSCAN with default params")
                dbscan = DBSCAN()

            labels_dbscan = dbscan.fit_predict(X)
            end_db = time.time() - start

            # Convert labels to list of sets
            partition_dbscan = labels_to_partition(labels_dbscan)
            dict_cluster['dbscan'][rho] = partition_dbscan

            # Spectral Clustering
            start = time.time()
            optimal_k, eigenvectors = find_optimal_clusters(G, plot)
            if nx.is_connected(G):
                # Use scikit-learn method
                clustering = SpectralClustering(n_clusters=optimal_k, affinity='precomputed', assign_labels='cluster_qr')
                labels_spectral = clustering.fit_predict(X)
            else:
                labels_spectral = compute_spectral_clustering(eigenvectors, optimal_k, method='kmeans', plot=False)
            end_spec = time.time() - start

            # Convert labels to list of sets
            partition_spectral = labels_to_partition(labels_spectral)

            dict_cluster['spectral'][rho] = partition_spectral

            # print(f"For rho {rho}: ")

            # print(f"Number of louvain cluster is {len(partition_louvain)}")
            # print(f"Number of leiden cluster is {len(partition_leiden)}")
            # print(f"Number of DBSCAN cluster is {len(partition_dbscan)}")
            # print(f"Number of Spectral cluster is {len(partition_spectral)}\n")

            # print(f"Time for Louvain -> {round(end_louv, 2)} s")
            # print(f"Time for Leiden -> {round(end_leiden, 2)} s")
            # print(f"Time for DBSCAN -> {round(end_db, 2)} s")
            # print(f"Time for Spectral -> {round(end_spec, 2)} s\n")
    
    return dict_cluster


def clustering_2_communities(results_squic, squic_method, method='scikit-learn'):
    '''
    if method='scikit-learn' use spectral clustering of this lib
    else use spectral clustering implemented in spectral_clustering.py file
    '''

    dict_cluster = {
        squic_method : {}
    }

    for technique, matrices in results_squic.items():
        for rho, X in matrices.items():
            if rho != 0.9997:
                # Ensure all off-diagonal entries are positive
                X = np.abs(X) 

                # Ensure diagonal entries are zero
                X.setdiag(0) # SQUIC_Fit ensure that, SQUIC not, so manually turn into 0

                # Create a graph from matrix X
                G = nx.from_scipy_sparse_array(X)

                print(f"\nfor rho {rho} graph is connected: {nx.is_connected(G)}")

                print(f"Computing spectral clustering for rho {rho}...")
                start = time.time()

                # eigenvalues, eigenvectors, degrees = debug_spectral_clustering(X)

                # print(f"for rho {rho} eigenvalues: {eigenvalues}")
                # print(f"for rho {rho} eigenvectors: {eigenvectors}")
                # print(f"for rho {rho} degrees: {degrees}")

                # X, _ = remove_isolated_nodes(X)

                if nx.is_connected(G): # if method == 'scikit-learn'
                    clustering = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='cluster_qr')
                    labels_spectral = clustering.fit_predict(X)
                # elif method == 'implemented':
                else:
                    # diagnose_adjacency_matrix(X)
                    L_norm = compute_normalized_laplacian(X) # fast
                    _, eigenvectors = compute_eigenvalues_eigenvectors(L_norm, k=2) # fast
                    labels_spectral = compute_spectral_clustering(eigenvectors, 2, method='kmeans') # fast
                
                end = time.time() - start

                print(f"For rho {rho} takes {end} seconds")
                # Convert labels to list of sets
                partition_spectral = labels_to_partition(labels_spectral)

                dict_cluster[technique][rho] = partition_spectral

    return dict_cluster


def internal_metrics(dict_cluster, W_matrices, leiden=False):
    matrix_dict = next(iter(W_matrices.values()))

    if not leiden:
        int_metrics = {
            rho: {
                'louvain': {},
                'dbscan': {},
                'spectral': {}
            } for rho in matrix_dict.keys()
        }
    else:
        int_metrics = {
            rho: {
                'louvain': {},
                'leiden': {},
                'dbscan': {},
                'spectral': {}
            } for rho in matrix_dict.keys()
        }

    for method, clustering_results in dict_cluster.items():
        for _, matrix in W_matrices.items():
            for l, X in matrix.items(): 
                partition = clustering_results[l]

                # Ensure all off-diagonal entries are positive
                X = np.abs(X) 

                # Ensure diagonal entries are zero
                X.setdiag(0) # SQUIC_Fit ensure that, SQUIC not, so manually turn into 0

                # G = nx.from_numpy_array(X)
                G = nx.from_scipy_sparse_array(X)

                node_to_community = {}
                for idx, comm in enumerate(partition):
                    for node in comm:
                        node_to_community[node] = idx

                n_cut = 0
                r_cut = 0

                clusters = set(node_to_community.values()) # set of clusters

                for cluster_id in clusters:
                    # Nodes in this cluster
                    cluster_nodes = [n for n, c in node_to_community.items() if c == cluster_id]
                    
                    # Nodes not in this cluster
                    other_nodes = [n for n in G.nodes() if n not in cluster_nodes]
                    
                    # Calculate cut: sum of weights between cluster and rest of graph
                    cut = nx.cut_size(G, cluster_nodes, other_nodes, weight='weight')
                    
                    # Calculate size of cluster
                    size = len(cluster_nodes)

                    # Calculate volume of cluster (sum of weights of edges connected to nodes in cluster)
                    volume = sum(dict(G.degree(cluster_nodes, weight='weight')).values())

                    # Normalized cut and ratio cut
                    r_cut += cut / size if size > 0 else 0
                    n_cut += cut / volume if volume > 0 else 0

                int_metrics[l][method]["ncut"] = float(round(n_cut, 2))
                int_metrics[l][method]["rcut"] = float(round(r_cut, 2))

                modularity = community.modularity(G, dict_cluster[method][l])
                int_metrics[l][method]['modularity'] = float(round(modularity, 2))
                
                # Connected Components
                int_metrics[l][method]['CC'] = nx.number_connected_components(G)

                # N cluster
                int_metrics[l][method]['nCluster'] = len(partition)

    return int_metrics


def modularity_fscore(dict_cluster, results_squic, account_prop):
    metrics = {
        method: {
            rho: {
                'spectral': {}
            } for rho in results_squic[method].keys()
        } for method in dict_cluster.keys()
    }

    n = len(account_prop)
    class_labels = np.array([account_prop[i]["class"] for i in range(n)])

    # Convert class labels to 0/1
    le = LabelEncoder()
    true_labels = le.fit_transform(class_labels)

    for method in dict_cluster:
        for rho, partition in dict_cluster[method].items():

            X = results_squic[method][rho]

            # Ensure matrix is in proper form
            X = np.abs(X)  # Make positive
            X.setdiag(0)   # Zero out diagonal

            G = nx.from_scipy_sparse_array(X)

            # Convert partition (list of sets) to labels
            cluster_labels = partition_to_labels(partition, n)

            # Compute modularity
            mod = community.modularity(G, partition)
            metrics[method][rho]['spectral']['modularity'] = round(float(mod), 2)

            # Number of clusters
            metrics[method][rho]['spectral']['nCluster'] = len(partition)

            # Compute F1-score (test both label alignments)
            f1a = f1_score(true_labels, 1 - cluster_labels, average='weighted')
            f1b = f1_score(true_labels, cluster_labels, average='weighted')

            f1 = max(f1a, f1b)

            metrics[method][rho]['spectral']['f1'] = round(f1, 2)

    return metrics


def ARI_fscore(dict_cluster, results_squic, account_prop):
    metrics = {
        method: {
            rho: {
                'spectral': {}
            } for rho in results_squic[method].keys()
        } for method in dict_cluster.keys()
    }

    n = len(account_prop)
    class_labels = np.array([account_prop[i]["class"] for i in range(n)])

    # Convert class labels to 0/1
    le = LabelEncoder()
    true_labels = le.fit_transform(class_labels)

    for method in dict_cluster:
        for rho, partition in dict_cluster[method].items():

            X = results_squic[method][rho]

            # Ensure matrix is in proper form
            X = np.abs(X)  # Make positive
            X.setdiag(0)   # Zero out diagonal

            G = nx.from_scipy_sparse_array(X)

            # Convert partition (list of sets) to labels
            cluster_labels = partition_to_labels(partition, n)

            # print(f"\nfor rho {rho}: true labels and then cluster labels")
            # print(true_labels)
            # print(cluster_labels)

            # Number of clusters
            metrics[method][rho]['spectral']['nCluster'] = len(partition)

            # Compute ARI
            ari = adjusted_rand_score(true_labels, cluster_labels)
            metrics[method][rho]['spectral']['ARI'] = round(ari, 2)

            # Compute F1 score (test both label alignments)
            f1a = f1_score(true_labels, 1 - cluster_labels, average='weighted')
            f1b = f1_score(true_labels, cluster_labels, average='weighted')

            # Choose the F1 score that is closer to ARI (normalized between 0:1)
            normalized_ari = (ari + 1)/2
            if abs(f1a - normalized_ari) < abs(f1b - normalized_ari):
                f1 = f1a
            else:
                f1 = f1b

            metrics[method][rho]['spectral']['f1'] = round(f1, 2)

    return metrics


def plot_Q_f1(metrics_dict):
    methods = ['squic-fit-matrix']
    colors = {
        'squic-fit-matrix': 'tab:orange'
    }

    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax2 = ax1.twinx()

    for method in methods:

        rhos = sorted(metrics_dict[method].keys())
        Q_values = []
        F1_values = []
        valid_rhos = []

        for rho in rhos:
            metrics = metrics_dict[method][rho].get('spectral', {})
            Q = metrics.get('modularity', None)
            f1 = metrics.get('f1', None)
            Q_values.append(Q)
            F1_values.append(f1)
            valid_rhos.append(rho)

        color = colors[method]

        # Modularity Q — dotted line
        ax1.plot(valid_rhos, Q_values, linestyle='dashed', marker='o', color='orange', label=f'Q')

        # F1-score — solid line
        ax2.plot(valid_rhos, F1_values, linestyle='solid', marker='^', color='green', label=f'F1')

    # Axis labels
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("Modularity Q", color='black')
    ax2.set_ylabel("F1 Score", color='black')

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    fig.tight_layout()
    plt.grid(True)
    plt.show()


def plot_ARI_f1(metrics_dict, squic_method, dimension, save=False):
    methods = [squic_method]
    colors = {
        squic_method: 'tab:orange'
    }

    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax2 = ax1.twinx()

    for method in methods:

        rhos = sorted(metrics_dict[method].keys())
        ARI_values = []
        F1_values = []
        valid_rhos = []

        for rho in rhos:
            metrics = metrics_dict[method][rho].get('spectral', {})
            ari = metrics.get('ARI', None)
            f1 = metrics.get('f1', None)
            ARI_values.append(ari)
            F1_values.append(f1)
            valid_rhos.append(rho)

        color = colors[method]

        # Modularity Q — dotted line
        ax1.plot(valid_rhos, ARI_values, linestyle='dashed', marker='o', color='orange', label=f'ARI')

        # F1-score — solid line
        ax2.plot(valid_rhos, F1_values, linestyle='solid', marker='^', color='green', label=f'F1')

    # Axis labels
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("ARI", color='black')
    ax2.set_ylabel("F1 Score", color='black')

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    fig.tight_layout()
    plt.grid(True)
    if save:
        # if path does not exists, create it
        os.makedirs(f'images/{dimension}', exist_ok=True)
        plt.savefig(f"images/{dimension}/ARI_F1_{squic_method}")
        print(f"Plot saved in images/{dimension}/ARI_F1_{squic_method}")
    plt.show()
