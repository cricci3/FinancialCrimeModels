import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import networkx as nx
from functions.SQUIC_functions import squic_fit_sparse, check_symmetric_sparse, read_lambdas
from networkx.algorithms import community
import igraph as ig
import leidenalg as la
from sklearn.cluster import DBSCAN
from functions.spectral_clustering import find_optimal_clusters, compute_spectral_clustering, compute_normalized_laplacian, compute_eigenvalues_eigenvectors
from functions.internal import load_dataset
from functions.plots import plot_timeseries, plot_PDens_Q_Times, plot_theta
import os
from functions.clustering_functions import labels_to_partition, compute_partition_density


if __name__ == '__main__':
    Y, name, dim, _ = load_dataset()

    PLOT = True
    SAVE_FIG = True
    if SAVE_FIG:
        SAVE_FIG_DIR = f'decomposition/glasso_images/{name.lower()}'
    
    if PLOT:
        plot_timeseries(Y, name, dimension=dim, save_fig=SAVE_FIG_DIR)

    all_residuals = { 
                account: seasonal_decompose(Y[account], model='additive', period=7).resid
                for account in Y.columns
            }
    df_residuals = pd.DataFrame(all_residuals).dropna()
    print(f"Data with {df_residuals.shape[0]} samples after decomposition.")

    print("-> Standardizing (normalizing) data...")
    residuals_matrix = df_residuals.values
    col_std = np.std(residuals_matrix, axis=0, keepdims=True)
    col_std[col_std == 0] = 1.0
    prepared_data = residuals_matrix / col_std

    df_prepared = pd.DataFrame(prepared_data, 
                            index=df_residuals.index, 
                            columns=df_residuals.columns)
    if PLOT:
        plot_timeseries(df_prepared, "AMLSIM", dimension=dim, save_fig=SAVE_FIG_DIR)

    print(df_prepared)

    prepared_data = prepared_data.T
    ROWS = prepared_data.shape[0]

    lambdas = read_lambdas(name, dim, "no-bias")
    
    int_metrics = {
        rho: {
            'louvain': {},
            'leiden': {},
            'dbscan': {},
            'spectral': {}
        } for rho in lambdas
    }
    
    for LAMBDA in lambdas:
        # Run SQUIC-Fit
        print(f"-> Running SQUIC-Fit with lambda = {LAMBDA}...")
        theta, end_time = squic_fit_sparse(prepared_data, LAMBDA, LAMBDA/10)
        end_time = round(end_time, 2)
        print(f"required time: {end_time}")

        int_metrics[LAMBDA]['time'] = end_time

        nnz = theta.count_nonzero()
        print(f"nnz = {int(nnz)} per rows = {round(int(nnz)/ROWS, 2)}")

        if check_symmetric_sparse(theta):
            print(f" Matrix is symmetric per lambda={LAMBDA}")
        else:
            print(f" Matrix is not symmetric per lambda={LAMBDA}")

        if PLOT:
            plot_theta(theta, name, dim, LAMBDA, SAVE_FIG)

        dict_cluster = {
            "louvain" : {},
            "spectral" : {},
            "dbscan" : {},
            "leiden" : {}
        }

        dbscan_params_dict = {
            '100' : {'epsilon' : 0.7,
                    'min_samples' : 8},
            '1K' : {'epsilon' : 0.7,
                    'min_samples' : 3},
            '10K' : {'epsilon' : 0.7,
                    'min_samples' : 3},
            '100K' : {'epsilon' : 0.7,
                    'min_samples' : 3},
            '1M' : {'epsilon' : 0.7,
                    'min_samples' : 3}
        }

        # Ensure all off-diagonal entries are positive
        X = np.abs(theta) 
        # Ensure diagonal entries are zero
        X.setdiag(0) # SQUIC_Fit ensure that

        # Create a graph from matrix X
        G = nx.from_scipy_sparse_array(X)

        connected = nx.is_connected(G)
        print(f"Is G connected for l={LAMBDA}? {connected}")
        if not connected:
            connected_components = nx.connected_components(G)
            component_sizes = [len(c) for c in connected_components]
            print(f"# CC : {len(component_sizes)}")

        # Louvain
        partition_louvain = community.louvain_communities(G)
        dict_cluster['louvain'] = partition_louvain

        # Leiden
        G_igraph = ig.Graph.from_networkx(G)
        partition_leiden = la.find_partition(G_igraph, la.ModularityVertexPartition)
        dict_cluster['leiden'] = partition_leiden

        dbscan = DBSCAN(eps=dbscan_params_dict[dim]['epsilon'],
                            min_samples=dbscan_params_dict[dim]['min_samples'],
                            metric='cosine')
        labels_dbscan = dbscan.fit_predict(X)

        # Convert labels to list of sets
        partition_dbscan = labels_to_partition(labels_dbscan)
        dict_cluster['dbscan'] = partition_dbscan

        # Spectral Clustering
        optimal_k, eigenvectors = find_optimal_clusters(G, dim, plot=False)
        if eigenvectors is not None:
            labels_spectral = compute_spectral_clustering(eigenvectors, optimal_k, method='kmeans', plot=False)
            # Convert labels to list of sets
            partition_spectral = labels_to_partition(labels_spectral)

            dict_cluster['spectral'] = partition_spectral
        else:
            dict_cluster['spectral'] = None


        for method, partition in dict_cluster.items():
            print(method)

            # Skip if partition is None
            if partition is None:
                print(f"[Warning] No partition for method={method}, lambda={LAMBDA}. Skipping.")

                int_metrics[LAMBDA][method]["p_density"] = 0
                int_metrics[LAMBDA][method]['modularity'] = 0
                int_metrics[LAMBDA][method]['CC'] = 0
                int_metrics[LAMBDA][method]['nCluster'] = 0
                int_metrics[LAMBDA][method]['isolated'] = "all isolated elements"
            else:
                m_total = G.number_of_edges()
                D_sum = 0
                isolated_node = 0
                density_sum = 0

                for comm in partition:
                    n = len(comm)
                    if n == 1:
                        isolated_node += 1
                        continue
                    else:
                        if n >= 3: # skip communities too small for partition density
                            D_sum = compute_partition_density(G, comm, D_sum)
                
                partition_density = (2 / m_total) * D_sum if m_total > 0 else 0
                int_metrics[LAMBDA][method]["p_density"] = float(round(partition_density, 2))

                modularity = community.modularity(G, partition)
                int_metrics[LAMBDA][method]['modularity'] = float(round(modularity, 2))
                int_metrics[LAMBDA][method]["isolated"] = isolated_node
                int_metrics[LAMBDA][method]['CC'] = nx.number_connected_components(G)
                int_metrics[LAMBDA][method]['nCluster'] = len(partition)
    
    for rho, results in int_metrics.items():
        print()
        print(f"Lambda = {rho}")
        for method, metrics in results.items():
            if method == 'time':
                pass
            else:
                print(f"    {method} : #Cluster = {metrics['nCluster']}, p_density = {metrics['p_density']}, Q = {metrics['modularity']}")


    if PLOT:
        plot_PDens_Q_Times(int_metrics, dim, name, save=SAVE_FIG)
