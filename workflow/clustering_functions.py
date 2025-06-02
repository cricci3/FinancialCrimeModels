import numpy as np

import networkx as nx
from networkx.algorithms import community
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from tqdm import tqdm
import time

from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict

from workflow.SQUIC_functions import print_matrix
import matplotlib.pyplot as plt
from workflow.spectral_clustering import find_optimal_clusters, compute_spectral_clustering
import igraph as ig
import leidenalg as la


def connect_isolated(X, min_correlation=1e-6):
    """
    Connect isolated nodes using very weak correlations
    
    Args:
        X: SQUIC correlation matrix
        min_correlation: minimum correlation to add for isolated nodes
    
    Returns:
        X_connected: connected adjacency matrix
    """
    
    G = nx.from_scipy_sparse_array(X)
    
    # Find connected components
    components = list(nx.connected_components(G))
    main_component = max(components, key=len)
    isolated_nodes = [list(comp)[0] for comp in components if len(comp) == 1]
    
    # print(f"Main component: {len(main_component)} nodes")
    # print(f"Isolated nodes: {len(isolated_nodes)} nodes")
    # print(f"Other components: {len(components) - len(isolated_nodes) - 1}")
    
    X_connected = X.copy()
        
    for isolated_node in isolated_nodes:
        # Connect to nearest node in main component (arbitrary but deterministic)
        main_nodes = list(main_component)
        # Use node index distance as a simple heuristic
        nearest_main = min(main_nodes, key=lambda x: abs(x - isolated_node))
        
        X_connected[isolated_node, nearest_main] = min_correlation
        X_connected[nearest_main, isolated_node] = min_correlation
        
        # print(f"  Weakly connected node {isolated_node} to node {nearest_main}")
    
    # Handle any remaining multi-node components
    remaining_components = [comp for comp in components 
                          if len(comp) > 1 and comp != main_component]
    
    for comp in remaining_components:
        # Connect each remaining component to main component
        comp_nodes = list(comp)
        main_nodes = list(main_component)
        
        # Find best existing connection or create weak one
        max_weight = 0
        best_edge = None
        
        for i in comp_nodes:
            for j in main_nodes:
                weight = max(X[i, j], X[j, i])
                if weight > max_weight:
                    max_weight = weight
                    best_edge = (i, j)
        
        if best_edge and max_weight > 0:
            X_connected[best_edge[0], best_edge[1]] = max_weight
            X_connected[best_edge[1], best_edge[0]] = max_weight
        else:
            # Create weak connection
            i, j = comp_nodes[0], main_nodes[0]
            X_connected[i, j] = min_correlation
            X_connected[j, i] = min_correlation
        
        # print(f"  Connected component of {len(comp)} nodes to main component")
    
    return X_connected


def connect_components_softly(X):
    X = X.tolil()

    n_components, labels = connected_components(csgraph=X, directed=False, return_labels=True)
    if n_components <= 1:
        return X.tocsr()
    
    # Identify isolated nodes (degree 0)
    isolated_nodes = [i for i in range(X.shape[0]) if X.rows[i] == []]

    if not isolated_nodes:
        return X.tocsr()

    # Find minimum non-zero value for weak connection
    min_val = X.data[0][0]
    for row in X.data:
        if row:
            min_val = min(min_val, min(row))
    min_val = min_val if min_val > 0 else 1e-6

    # For each isolated node, find its most similar node (based on X row)
    for node in isolated_nodes:
        # Search in the full matrix (sparse row)
        row = X.getrow(node).tocoo()
        # fallback: connect to the first non-isolated node with nonzero connection
        best_score = -1
        best_neighbor = None
        for j in range(X.shape[0]):
            if j == node or j in isolated_nodes:
                continue
            score = X[j, node]
            if score > best_score:
                best_score = score
                best_neighbor = j
        if best_neighbor is not None:
            X[node, best_neighbor] = min_val
            X[best_neighbor, node] = min_val

    return X.tocsr()


def similarity_graph(G, account_prop, X):
    '''
    Use the "class" from account_prop to build a class-based similarity matrix and then merge with matrix
    '''
    
    print("Using Similairty graph")

    components = list(nx.connected_components(G))
    main_component = max(components, key=len)
    isolated_nodes = [list(comp)[0] for comp in components if len(comp) == 1]

    print(f"Main component: {len(main_component)} nodes")
    print(f"Isolated nodes: {len(isolated_nodes)} nodes")
    print(f"Other components: {len(components) - len(isolated_nodes) - 1}")

    n = len(account_prop)
    class_labels = np.array([account_prop[i]["class"] for i in range(n)])

    W_sim = lil_matrix((n, n))  # initialize Similarity graph as sparse matrix

    # Group indices by class
    class_to_indices = defaultdict(list)
    for i, cls in enumerate(class_labels):
        class_to_indices[cls].append(i)

    # For each class, connect all users in that class (like block diagonal)
    for indices in class_to_indices.values():
        for i in indices:
            W_sim[i, indices] = 1.0  # Vectorized row update

    W_sim = W_sim.tocsr()

    # Fuse X with sim graph
    alpha = 0.8

    # W-fused[i,j]= a⋅W-squic[i,j]+(1−α)⋅Wsim[i,j]
    X = alpha * X + (1 - alpha) * W_sim
    # If the users are strongly correlated statistically -> W_squic[i,j] will dominate
    # For disconnected or weakly connected nodes, if the users are of the same class (W_sim[i,j] = 1) will contribute

    # Normalize (unit max)
    X = X / np.max(X)

    return X


# def resolve_graph_connection(X, name, account_prop):
#     # if graph not connected, user Similairty Graph for paysim, other technique for AMLSIM/Libra
#     G = nx.from_scipy_sparse_array(X)

#     # if name != 'PAYSIM':
#     #     # while not nx.is_connected(G):
#     #     #     print(f"Graph not connected. {nx.number_connected_components(G)} components found.")

#     #     #     X = connect_isolated(X)
#     #     #     G = nx.from_scipy_sparse_array(X)

#     #     #     print(f"Is connected? {nx.is_connected(G)}")
#     #     X = connect_components_softly(X)

#     # else: # for PAYSIM only -> use similarity graph
#     #     # print(f"Graph not connected. {nx.number_connected_components(G)} components found.")

#     #     # X = similarity_graph(G, account_prop, X)
#     #     # G = nx.from_scipy_sparse_array(X)

#     #     # if not nx.is_connected(G):
#     #     #     while not nx.is_connected(G):
#     #     #         print("Graph still not connected after Similarity Graph")
#     #     #         X = connect_isolated(X)
#     #     #         G = nx.from_scipy_sparse_array(X)
#     #     #     print("Graph connected")
#     #     # else:
#     #     #     print("Graph connected after Similarity Graph")
#         # X = connect_components_softly(X)
#     X = connect_components_softly(X)
#     G = nx.from_scipy_sparse_array(X)

#     return X, G


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

        # if nx.is_connected(G):
        #     print("Graph already connected!")
        # else:
        #     X, G = resolve_graph_connection(X, name, account_prop)

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

def clustering_optimal_number(W_matrices):
    dict_cluster = {
        "louvain" : {},
        "spectral" : {},
        "dbscan" : {},
        "leiden" : {}
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

        # Use n_cluster = len(partitionin louvain)
        start = time.time()
        optimal_k, eigenvectors = find_optimal_clusters(G, plot=False)
        labels_spectral = compute_spectral_clustering(eigenvectors, optimal_k, method='hierarchical', plot=False)
        end_spec = time.time() - start

        # Convert labels to list of sets
        partition_spectral = labels_to_partition(labels_spectral)

        dict_cluster['spectral'][rho] = partition_spectral

        print(f"For rho {rho}: ")

        print(f"Number of louvain cluster is {len(partition_louvain)}")
        print(f"Number of leiden cluster is {len(partition_leiden)}")
        print(f"Number of DBSCAN cluster is {len(partition_dbscan)}")
        print(f"Number of Spectral cluster is {len(partition_spectral)}\n")

        print(f"Time for Louvain -> {end_louv}")
        print(f"Time for Leiden -> {end_leiden}")
        print(f"Time for DBSCAN -> {end_db}")
        print(f"Time for Spectral -> {end_spec}\n")
    
    return dict_cluster


def clustering_2_communities(results_squic, name, account_prop):
    dict_cluster = {
        "squic-fit-matrix" : {}
    }

    for technique, matrices in results_squic.items():
        for rho, X in matrices.items():
            # Ensure all off-diagonal entries are positive
            X = np.abs(X) 

            # Ensure diagonal entries are zero
            X.setdiag(0) # SQUIC_Fit ensure that, SQUIC not, so manually turn into 0

            # Create a graph from matrix X
            G = nx.from_scipy_sparse_array(X)

            # while not nx.is_connected(G):
            #     print(f"for rho {rho} graph is connected {nx.is_connected(G)}")
            #     X = connect_components_softly(X)            
            #     G = nx.from_scipy_sparse_array(X)
            print(f"for rho {rho} graph is connected {nx.is_connected(G)}")

            # while not nx.is_connected(G):
            #     print(f"For rho {rho} graph not connected")
            #     components = list(nx.connected_components(G))
            #     main_component = max(components, key=len)
            #     isolated_nodes = [list(comp)[0] for comp in components if len(comp) == 1]
                
            #     # print(f"Main component: {len(main_component)} nodes")
            #     # print(f"Isolated nodes: {len(isolated_nodes)} nodes")
            #     # print(f"Other components: {len(components) - len(isolated_nodes) - 1}\n")
                
            #     # if len(isolated_nodes) > 0:
            #     #     print(isolated_nodes)
            #     #     print()

            #     G = nx.from_scipy_sparse_array(X)

            # print(f"For rho {rho}, now graph is connected")

            start = time.time()
            # assign labels: The strategy to use to assign labels in the embedding space.
            # There are three ways to assign labels after the Laplacian embedding.
            # - k-means can be applied and is a popular choice. But it can also be sensitive to initialization.
            # - Discretization is another approach which is less sensitive to random initialization
            # - The cluster_qr method directly extracts clusters from eigenvectors in spectral clustering.
            #      In contrast to k-means and discretization, cluster_qr has no tuning parameters and is not an iterative method,
            #      yet may outperform k-means and discretization in terms of both quality and speed.
            clustering = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='cluster_qr')
            labels_spectral = clustering.fit_predict(X)

            end_spec = time.time() - start

            # Convert labels to list of sets
            partition_spectral = labels_to_partition(labels_spectral)
            dict_cluster[technique][rho] = partition_spectral

    return dict_cluster


def internal_metrics(dict_cluster, W_matrices, leiden=False):
    if leiden == False:
        int_metrics = {
            rho: {
                'louvain' : {},
                'dbscan' : {},
                'spectral' : {}
            } for rho in W_matrices.keys()
        }
    else:
        int_metrics = {
            rho: {
                'louvain' : {},
                'leiden' : {},
                'dbscan' : {},
                'spectral' : {},
            } for rho in W_matrices.keys()
        }

    for method, clustering_results in dict_cluster.items():
        for l, X in W_matrices.items():
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
            metrics[method][rho]['spectral']['modularity'] = round(float(mod), 4)

            # Number of clusters
            metrics[method][rho]['spectral']['nCluster'] = len(partition)

            # Compute F1-score (test both label alignments)
            f1a = f1_score(true_labels, 1 - cluster_labels, average='weighted')
            f1b = f1_score(true_labels, cluster_labels, average='weighted')

            f1 = max(f1a, f1b)

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
