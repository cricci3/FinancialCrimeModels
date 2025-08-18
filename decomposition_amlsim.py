import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.covariance import GraphicalLasso
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from functions.SQUIC_functions import squic_fit_sparse, compute_squic, nnz_sparse, check_symmetric_sparse
from functions.clustering_functions import clustering_optimal_number, modularity_density
from networkx.algorithms import community
import igraph as ig
import leidenalg as la
from sklearn.cluster import DBSCAN, SpectralClustering
from functions.spectral_clustering import find_optimal_clusters, compute_spectral_clustering, compute_normalized_laplacian, compute_eigenvalues_eigenvectors
from functions.internal import load_dataset_1B


def internal_density(G, community, n, density_sum):
    subgraph = G.subgraph(community)
    m = subgraph.number_of_edges()
    max_possible = n * (n - 1) / 2
    density_sum += m / max_possible # internal density of single comm
    return density_sum


DIR = 'glasso_images'
PLOT = False

# SEASONAL_PERIOD = 1 to disable this step if no cycles are suspected.
SEASONAL_PERIOD = 7

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

    Y, _, dim, _ = load_dataset_1B()

    import matplotlib.pyplot as plt

    print("-> Plotting line graph of all processed account time series...")
    plt.figure(figsize=(16, 8))
    plt.plot(Y.index, Y, color='blue', alpha=0.5, linewidth=0.7)
    plt.title('All accounts time series after decomposition and normalization', fontsize=16)
    plt.xlabel('Time step (day)')
    plt.ylabel('Balance')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--', linewidth=1) # Add a line at zero
    plt.show()

    from statsmodels.tsa.seasonal import seasonal_decompose
    import pandas as pd

    SEASONAL_PERIOD = 7

    all_residuals = { 
                account: seasonal_decompose(Y[account], model='additive', period=SEASONAL_PERIOD).resid
                for account in Y.columns
            }
    df_residuals = pd.DataFrame(all_residuals).dropna()
    print(f"Data with {df_residuals.shape[0]} samples after decomposition.")

    import numpy as np

    print("-> Standardizing (normalizing) data...")
    residuals_matrix = df_residuals.values
    col_std = np.std(residuals_matrix, axis=0, keepdims=True)
    col_std[col_std == 0] = 1.0
    prepared_data = residuals_matrix / col_std

    print("-> Plotting line graph of all processed account time series...")
    plt.figure(figsize=(16, 8))
    plt.plot(df_residuals.index, prepared_data, color='blue', alpha=0.1, linewidth=0.7)
    plt.title('All accounts time series after decomposition and normalization', fontsize=16)
    plt.xlabel('Time step (day)')
    plt.ylabel('Standardized residual value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--', linewidth=1) # Add a line at zero
    # plt.savefig(ALL_SERIES_PLOT_FILE, dpi=300, bbox_inches='tight')
    plt.show()

    if dim == '100':
        ROWS = 100
    elif dim == '1K':
        ROWS = 1000
    elif dim == '10K':
        ROWS = 10000
    elif dim == '100K':
        ROWS = 100000

    print(prepared_data.shape)
    prepared_data = prepared_data.T

    print(prepared_data.shape)

    # reg. parameter glasso
    if dim == '100':
        lambdas = [0.4, 0.3, 0.2, 0.1, 0.09, 0.07, 0.05, 0.02]
        # LAMBDA = 0.05
    elif dim == '1K':
        lambdas = [0.999, 0.95, 0.7, 0.5, 0.3]
        # lambdas = [0.9995, 0.999, 0.95, 0.9, 0.7, 0.5]
        # LAMBDA = 0.5
    elif dim == '10K':
        # lambdas = [0.9999995]
        lambdas = [0.999995, 0.999, 0.995, 0.8]
        # LAMBDA = 0.99
    elif dim == '100K':
        # lambdas = [0.995]
        lambdas = [0.9999999, 0.9999995, 0.99999, 0.99995, 0.9999, 0.9995, 0.999]
    
    int_metrics = {
        rho: {
            'louvain': {},
            'leiden': {},
            'dbscan': {},
            'spectral': {}
        } for rho in lambdas
    }
    
    print(int_metrics)

    for LAMBDA in lambdas:
        # Run SQUIC-Fit ---
        print(f"-> Running SQUIC-Fit with lambda = {LAMBDA}...")
        model_fit, end_time = squic_fit_sparse(prepared_data, LAMBDA, LAMBDA/10)
        end_time = round(end_time, 2)
        print(f"required time: {end_time}")

        nnz = model_fit.count_nonzero()
        print(f"nnz = {int(nnz)} per rows = {int(nnz)/ROWS}")


        if check_symmetric_sparse(model_fit):
            print(f" Matrix is symmetric per rho {LAMBDA}")
        else:
            print(f" Matrix is not symmetric per rho {LAMBDA}")

        plt.figure(figsize=(7, 7))
        plt.spy(model_fit, markersize=5)
        plt.ylabel("Users", fontsize=18)

        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)  
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

        # for method, results in int_metrics.items():
        #     print(f"    {method}: nCluster = {results['nCluster']}, CC = {results['CC']}, nIsolated  = {results['isolated']}, IntDensity = {results['int_density']}, PDensity = {results['p_density']}, Q = {results['modularity']}")

    
    for rho, results in int_metrics.items():
        print(f"Lambda = {rho}")
        for method, metrics in results.items():
            print(f"    {method} : #Cluster = {metrics['nCluster']}, int_density = {metrics['int_density']}, p_density = {metrics['p_density']}, Q = {metrics['modularity']}")


    colors = {
        'louvain':'green',
        'leiden':'orange',
        'spectral':'cornflowerblue',
        'dbscan':'mediumorchid'
    }

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    # Get the list of methods from the first rho entry
    first_rho = next(iter(int_metrics))
    clustering_methods = int_metrics[first_rho].keys()

    for method in clustering_methods:
        PDensity_values = []
        second_values = []
        valid_rhos = []

        for rho in sorted(int_metrics.keys()):
            method_metrics = int_metrics[rho].get(method, {})
            pdens = method_metrics.get('p_density', None)
            second_metric = method_metrics.get('modularity', None)


            if pdens is not None and second_metric is not None:
                PDensity_values.append(pdens)
                second_values.append(second_metric)
                valid_rhos.append(rho)

        if valid_rhos:
            color = colors.get(method, 'black')
            ax1.plot(valid_rhos, second_values, linestyle='dashed', marker='o', color=color, label=f'Q {method}')
            ax2.plot(valid_rhos, PDensity_values, linestyle='solid', marker='^', color=color, label=f'Pdensity {method}')

    # ax1.set_xlabel("Reg. Parameter lambda", fontsize=18)
    # plt.rcParams['text.usetex'] = True  # Enable LaTeX
    ax1.set_xlabel(r'Reg. parameter $\lambda$', fontsize=18)
    ax1.set_ylabel("Modularity Q", color='black', fontsize=18)
    ax2.set_ylabel("Partition Density", color='black', fontsize=18)

    ax1.set_xticks(valid_rhos)
    ax1.set_xticklabels([str(r) for r in valid_rhos], fontsize=18)

    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=12)

    fig.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
