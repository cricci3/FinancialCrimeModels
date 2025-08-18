import pandas as pd
import numpy as np
from sklearn.covariance import GraphicalLasso
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from functions.SQUIC_functions import squic_fit_sparse, squic_fit_matrix_sparse, check_symmetric_sparse
from functions.clustering_functions import clustering_optimal_number, modularity_density
from networkx.algorithms import community
import igraph as ig
import leidenalg as la
from sklearn.cluster import DBSCAN, SpectralClustering
from functions.spectral_clustering import find_optimal_clusters, compute_spectral_clustering, compute_normalized_laplacian, compute_eigenvalues_eigenvectors
from functions.internal import load_dataset_1A
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.neighbors import NearestNeighbors
import os
from matplotlib.gridspec import GridSpec
from functions.plots import plot_timeseries


PLOT = False
# SEASONAL_PERIOD = 1 to disable this step if no cycles are suspected.
SEASONAL_PERIOD = 7
SAVE_FIG = True


def internal_density(G, community, n, density_sum):
    subgraph = G.subgraph(community)
    m = subgraph.number_of_edges()
    max_possible = n * (n - 1) / 2
    density_sum += m / max_possible # internal density of single comm
    return density_sum


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


def compute_partition_density(G, community, D_sum):
    n_alpha = len(community)

    subgraph = G.subgraph(community)
    m_alpha = subgraph.number_of_edges()

    numerator = m_alpha - (n_alpha - 1)
    denominator = (n_alpha - 2) * (n_alpha - 1)

    if denominator > 0:
        D_sum += m_alpha * (numerator / denominator)
    return D_sum


def main():
    """
    Transaction logs -> conditional dependency graph of unexpected account activity.
    """

    if input("Do you want to load data?").upper() == 'Y':
        dim = input("Which dimension (100/1K/10K/100K/1M)?").upper()
        path = f'decomposition/paysim_data_saved/{dim}'
    
        # Load pre-saved Y_norm
        chunk_size = 10000
        chunks = []

        for chunk in pd.read_csv(f'{path}/YNorm_{dim}.csv', chunksize=chunk_size):
            chunks.append(chunk)

        prepared_data = pd.concat(chunks, ignore_index=True)

        print(prepared_data)
    else:
        Y, _, dim, account_prop = load_dataset_1A()

        plot_timeseries(Y, "PAYSIM", dimension=dim, print_fig=False)

        # plt.figure(figsize=(16, 8))
        # plt.plot(Y.index, Y, alpha=0.5, linewidth=0.7)
        # plt.title('All accounts time series', fontsize=16)
        # plt.xlabel('Time step (day)')
        # plt.ylabel('Balance')
        # plt.grid(True, linestyle='--', alpha=0.6)
        # plt.axhline(0, color='black', linestyle='--', linewidth=1)
        # if SAVE_FIG:
        #     plt.savefig(f'paysim_ts_{dim}', dpi=300)
        # plt.show()

        all_residuals = { 
                    account: seasonal_decompose(Y[account], model='additive', period=SEASONAL_PERIOD).resid
                    for account in Y.columns
                }
        df_residuals = pd.DataFrame(all_residuals).dropna()
        print(f"Data with {df_residuals.shape[0]} samples after decomposition.")


        print("-> Standardizing (normalizing) data...")
        residuals_matrix = df_residuals.values
        col_std = np.std(residuals_matrix, axis=0, keepdims=True)
        col_std[col_std == 0] = 1.0
        prepared_data = residuals_matrix / col_std

        print("-> Plotting line graph of all processed account time series...")
        plt.figure(figsize=(9,7))
        plt.plot(df_residuals.index, prepared_data, alpha=0.7)
        plt.axhline(0, color='black', linestyle='--', linewidth=1) # Add a line at zero
        plt.xlabel("Days", fontsize=18)
        plt.ylabel("Standardized residual value", fontsize=18)
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        plt.grid(axis='y', color='#cccccc', linewidth=0.5, alpha=0.3, linestyle='--')
        plt.grid(axis='x', color='#cccccc', linewidth=0.5, alpha=0.3, linestyle='--')
        plt.tight_layout()
        if SAVE_FIG:
            plt.savefig(f'ts_residual_PaySim_{dim}', dpi=300)
        plt.show()

        print(prepared_data.shape)
        prepared_data = prepared_data.T

        print(prepared_data.shape)

        if input("Do you want to cache this data?").upper() == 'Y':
            print("Saving data")
            path = f'decomposition/paysim_data_saved/{dim}'

            # if path does not exists, create it
            os.makedirs(path, exist_ok=True)
            
            # Save Y_norm as CSV
            pd.DataFrame(prepared_data).to_csv(f'{path}/YNorm_{dim}.csv', index=False)
    
    ROWS = prepared_data.shape[0]

    # reg. parameter glasso
    if dim == '100':
        lambdas = [0.6, 0.5, 0.2, 0.1, 0.09]
    elif dim == '1K':
        lambdas = [0.9, 0.7, 0.5, 0.3]
        # lambdas = [0.9995, 0.999, 0.95, 0.9, 0.7, 0.5]
    elif dim == '10K':
        lambdas = [0.995, 0.9, 0.8, 0.75, 0.7]
    elif dim == '100K':
        # lambdas = [0.99999, 0.99995, 0.9999]
        # lambdas = [0.9999, 0.9995, 0.999, 0.995, 0.99] # w/o bias
        lambdas = [0.99, 0.95, 0.9, 0.85] # w/o bias
    
    # # reg. parameter glasso
    # if dim == '100':
    #     lambdas = [0.4, 0.21, 0.2, 0.19, 0.1]
    #     # lambdas = [0.09, 0.07]
    # elif dim == '1K':
    #     # lambdas = [0.7, 0.5, 0.3] for residual
    #     lambdas = [0.7, 0.65, 0.6, 0.55, 0.5] # for net change
    # elif dim == '10K':
    #     # lambdas = [0.9, 0.8, 0.7] # residual
    #     lambdas = [0.9, 0.8, 0.75]
    # elif dim == '100K':
    #     # lambdas = [0.95, 0.92, 0.9] # without bias but not good
    #     lambdas = [0.9999999, 0.9999995] # bias
    
    int_metrics = {
        rho: {
            'louvain': {},
            'leiden': {},
            'dbscan': {},
            'spectral': {}
        } for rho in lambdas
    }

    bias = input("Do you want to run SQUIC-Fit with bias? (Y/N)").upper()
    if bias == 'Y':
        n_neigs = {
            '100' : 10,
            '1K' : 10,
            '10K' : 4, # before it was 4
            '100K' : 3, # can run even with 4
            '1M' : 2
        }

        nbrs = NearestNeighbors(n_neighbors=n_neigs[dim], metric='euclidean', n_jobs=-1)
        nbrs.fit(prepared_data)
        knn_matrix = nbrs.kneighbors_graph(prepared_data, mode='connectivity')
        
        print("Shape KNN:", knn_matrix.shape)
        print("nnz KNN:", knn_matrix.nnz)

        plt.figure(figsize=(7, 7))
        plt.spy(knn_matrix, markersize=5)
        plt.xlabel("Users", fontsize=18)
        plt.ylabel("Users", fontsize=18)
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        plt.title(f"KNN with K={n_neigs[dim]}")
        plt.show()

    for LAMBDA in lambdas:
        # Run SQUIC-Fit ---
        print(f"-> Running SQUIC-Fit with lambda = {LAMBDA}...")
        if bias == 'N':
            model_fit, end_time = squic_fit_sparse(prepared_data, LAMBDA, LAMBDA/10)
        else:
            model_fit, end_time = squic_fit_matrix_sparse(prepared_data, LAMBDA, knn_matrix)
    
        end_time = round(end_time, 2)
        print(f"required time: {end_time}")

        int_metrics[LAMBDA]['time'] = end_time

        nnz = model_fit.count_nonzero()
        print(f"nnz = {int(nnz)} per rows = {round(int(nnz)/ROWS, 2)}")

        if check_symmetric_sparse(model_fit):
            print(f" Matrix is symmetric per rho {LAMBDA}")
        else:
            print(f" Matrix is not symmetric per rho {LAMBDA}")

        plt.figure(figsize=(7, 7))
        plt.spy(model_fit, markersize=5)
        # plt.xlabel("Users", fontsize=18)
        plt.xlabel("Users", fontsize=18)
        plt.ylabel("Users", fontsize=18)

        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18) 
        if SAVE_FIG:
            path = f'decomposition/glasso_images/paysim/{dim}'
            os.makedirs(path, exist_ok=True)
            plt.savefig(f"{path}/squic_fit_{LAMBDA}", dpi=180)
        plt.show()


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
        X = np.abs(model_fit) 

        # Ensure diagonal entries are zero
        X.setdiag(0) # SQUIC_Fit ensure that, SQUIC not, so manually turn into 0

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
        optimal_k, eigenvectors = find_optimal_clusters(G, dim, plot=PLOT)
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
                # Ensure all off-diagonal entries are positive
                X = np.abs(X) 

                # Ensure diagonal entries are zero
                X.setdiag(0) # SQUIC_Fit ensure that, SQUIC not, so manually turn into 0

                # G = nx.from_numpy_array(X)
                G = nx.from_scipy_sparse_array(X)
                
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
                        density_sum = internal_density(G, comm, n, density_sum)

                        if n >= 3: # skip communities too small for partition density
                            D_sum = compute_partition_density(G, comm, D_sum)
                
                avg_internal_density = density_sum / len(partition) # avg internal density
                int_metrics[LAMBDA][method]["int_density"] = float(round(avg_internal_density, 2))

                partition_density = (2 / m_total) * D_sum if m_total > 0 else 0
                int_metrics[LAMBDA][method]["p_density"] = float(round(partition_density, 2))

                modularity = community.modularity(G, partition)
                int_metrics[LAMBDA][method]['modularity'] = float(round(modularity, 2))

                int_metrics[LAMBDA][method]["isolated"] = isolated_node
                
                # Connected Components
                int_metrics[LAMBDA][method]['CC'] = nx.number_connected_components(G)

                # N cluster
                int_metrics[LAMBDA][method]['nCluster'] = len(partition)

    print(int_metrics.items())
    
    for rho, results in int_metrics.items():
        print(f"Lambda = {rho}")
        print()
        for method, metrics in results.items():
            if method == 'time':
                pass
            else:
                print(f"    {method} : #Cluster = {metrics['nCluster']}, int_density = {metrics['int_density']}, p_density = {metrics['p_density']}, Q = {metrics['modularity']}")


    colors = {
        'louvain': 'green',
        'leiden': 'orange',
        'spectral': 'cornflowerblue',
        'dbscan': 'mediumorchid'
    }

    # figure with a short time panel below
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])          # Q (left y)
    ax2 = ax1.twinx()                         # Pdensity (right y)
    ax_time = fig.add_subplot(gs[1, 0], sharex=ax1)

    # Collect the sorted unique rho values (λ) for x (equally spaced categories)
    all_rhos = sorted(int_metrics.keys())
    x_pos_map = {rho: i for i, rho in enumerate(all_rhos)}  # rho -> index 0..N-1

    # Methods present in the first rho entry — keep only dict-valued keys (exclude 'time')
    first_rho = all_rhos[0]
    clustering_methods = [k for k, v in int_metrics[first_rho].items() if isinstance(v, dict)]

    # --- Time panel data (seconds) ---
    time_values, time_x = [], []
    for rho in all_rhos:
        end_time = int_metrics.get(rho, {}).get('time', np.nan)  # float seconds
        # if some entry is missing or NaN, keep NaN (line plot will skip; bar will show 0 if we choose to)
        time_values.append(end_time)
        time_x.append(x_pos_map[rho])

    # --- Top panel: Q and Pdensity for each method ---
    for method in clustering_methods:
        x_idx, q_vals, pden_vals = [], [], []
        for rho in all_rhos:
            mm = int_metrics.get(rho, {}).get(method, None)
            if not isinstance(mm, dict):
                continue
            pdens = mm.get('p_density', None)
            modularity = mm.get('modularity', None)
            if (pdens is not None) and (modularity is not None):
                x_idx.append(x_pos_map[rho])
                q_vals.append(modularity)
                pden_vals.append(pdens)

        if x_idx:
            c = colors.get(method, 'black')
            ax1.plot(x_idx, q_vals, linestyle='--', marker='o', color=c, label=f'Q {method}')
            ax2.plot(x_idx, pden_vals, linestyle='-', marker='^', color=c, label=f'Pdensity {method}')

    # ===== top axes labels & ticks =====
    ax1.set_ylabel('Modularity Q', color='black', fontsize=16)
    ax2.set_ylabel('Partition Density', color='black', fontsize=16)
    ax1.set_xticks(range(len(all_rhos)))
    ax1.set_xticklabels([str(r) for r in all_rhos], fontsize=13)
    ax1.tick_params(axis='y', labelsize=13)
    ax2.tick_params(axis='y', labelsize=13)
    ax1.grid(True, axis='y', alpha=0.3)

    # unified legend (top plot)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
            loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=4, fontsize=11, frameon=True)

    # ===== bottom time panel (seconds) =====
    time_values = np.array(time_values, dtype=float)
    # bars (replace NaN with 0 for bars; or switch to a line plot to skip NaNs)
    ax_time.bar(time_x, np.nan_to_num(time_values, nan=0.0),
                width=0.6, alpha=0.6, linewidth=0.5)

    ax_time.set_ylabel('Time (s)', fontsize=14)
    ax_time.set_xlabel(r'Reg. parameter $\lambda$', fontsize=16)
    ax_time.tick_params(axis='x', labelsize=13)
    ax_time.tick_params(axis='y', labelsize=12)
    ax_time.grid(True, axis='y', alpha=0.2)

    # hide duplicated x tick labels on top axis
    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.tight_layout()
    if SAVE_FIG:
        plt.savefig(f'paysim_{dim}', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
