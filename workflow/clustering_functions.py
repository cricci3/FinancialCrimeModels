import numpy as np

import networkx as nx
from networkx.algorithms import community
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from tqdm import tqdm
import time

from scipy.sparse import lil_matrix
from collections import defaultdict


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


def resolve_graph_connection(X, name, account_prop):
    # if graph not connected, user Similairty Graph for paysim, other technique for AMLSIM/Libra
    G = nx.from_scipy_sparse_array(X)

    if name != 'PAYSIM':
        while not nx.is_connected(G):
            print(f"Graph not connected. {nx.number_connected_components(G)} components found.")

            X = connect_isolated(X)
            G = nx.from_scipy_sparse_array(X)

            print(f"Is connected? {nx.is_connected(G)}")

    else: # for PAYSIM only -> use similarity graph
        print(f"Graph not connected. {nx.number_connected_components(G)} components found.")

        X = similarity_graph(G, account_prop, X)
        G = nx.from_scipy_sparse_array(X)

        if not nx.is_connected(G):
            while not nx.is_connected(G):
                print("Graph still not connected after Similarity Graph")
                X = connect_isolated(X)
                G = nx.from_scipy_sparse_array(X)
            print("Graph connected")
        else:
            print("Graph connected after Similarity Graph")

    return X, G


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


def clustering(W_matrices, name, account_prop):
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

        if nx.is_connected(G):
            print("Graph already connected!")
        else:
            X, G = resolve_graph_connection(X, name, account_prop)

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


def clustering_2_communities(results_squic, name, account_prop):
    dict_cluster = {
        "squic" : {},
        "squic-matrix" : {},
        "squic-fit" : {},
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

            if nx.is_connected(G):
                print("Graph already connected!")
            else:
                X, G = resolve_graph_connection(X, name, account_prop)

            start = time.time()
            clustering = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='cluster_qr')
            labels_spectral = clustering.fit_predict(X)

            end_spec = time.time() - start

            # Convert labels to list of sets
            partition_spectral = labels_to_partition(labels_spectral)
            dict_cluster[technique][rho] = partition_spectral

    return dict_cluster


def internal_metrics(dict_cluster, W_matrices):
    int_metrics = {
        rho: {
            'louvain' : {},
            'dbscan' : {},
            'spectral' : {}
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
            metrics[method][rho]['spectral']['modularity'] = float(mod)

            # Number of clusters
            metrics[method][rho]['spectral']['nCluster'] = len(partition)

            # Compute F1-score (test both label alignments)
            f1 = f1_score(true_labels, 1 - cluster_labels, average='weighted')

            metrics[method][rho]['spectral']['f1'] = f1

    return metrics
