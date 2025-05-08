from workflow.preprocess import *
from workflow.SQUIC_functions import *

import matplotlib.pyplot as plt
import matplotlib.colors as plt_color
import seaborn as sns

import json
import numpy as np

import networkx as nx
from networkx.algorithms import community
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score

from cosmograph import cosmo

from sklearn.metrics import silhouette_score
from tqdm import tqdm


def parse_input(user_input):
    """Parse and validate the dataset input in the format NAME_DIMENSION."""
    try:
        name, dimension = user_input.strip().upper().split("_")
    except ValueError:
        raise ValueError("Input must be in the format NAME_DIMENSION (e.g., AMLSIM_10K)")

    valid_names = {"AMLSIM", "PAYSIM", "LIBRE"}
    valid_dimensions = {"100", "1K", "10K", "100K", "1M"}

    if name not in valid_names:
        raise ValueError(f"Invalid dataset name '{name}'. Valid options are: {', '.join(valid_names)}")
    if dimension not in valid_dimensions:
        raise ValueError(f"Invalid dimension '{dimension}'. Valid options are: {', '.join(valid_dimensions)}")

    return name, dimension


def load_dataset():
    """Prompt user input and load the corresponding dataset."""
    while True:
        user_input = input("Insert dataset name in the following format NAME_DIMENSION (e.g., AMLSIM_10K): ")

        try:
            name, dimension = parse_input(user_input)
            break
        except ValueError as e:
            print(f"Error: {e}")
            continue

    if name == 'AMLSIM':
        df, account_prop = AMLSim_preprocessing(dimension)
    elif name == 'PAYSIM':
        df, account_prop = PaySim_preprocessing(dimension)
    elif name == 'LIBRE':
        # future implementation
        df = None
        print("LIBRE dataset support not yet implemented.")
    else:
        df = None

    return df, name, dimension, account_prop


def extract_timeseries(df, name):
    # Plot the balance evolution for all users (columns) as separate lines
    # plt.figure(figsize=(15,10), dpi= 300)
    plt.figure(figsize=(15,10))

    if name == 'AMLSIM':
        # Plot each column (account balance) as a line
        for user in df.columns:
            # color = 'mediumseagreen' if user.startswith('C') else 'hotpink'

            plt.plot(df.index, df[user], label=f"User {user}", alpha=0.6)

    elif name == 'PAYSIM':
        # Plot each column (account balance) as a line
        for user in df.columns:
            if user.startswith('C'):
                color = 'mediumseagreen'
            elif user.startswith('B'):
                color = 'crimson'
            else:
                color = 'darkturquoise'
            plt.plot(df.index, df[user], color=color, label=f"User {user}", alpha=0.6)

    # Add title and labels
    plt.xlabel("Days", fontsize=22)
    plt.ylabel("Balance", fontsize=22)

    # Rotate x-axis labels for better visibility
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)

    # Display the plot
    plt.tight_layout()
    plt.show()
    

def normalization(df, name):
    Y = df.T.to_numpy()  # Convert to (users, days)

    if name == 'AMLSIM':
        # normalize the Y input data to achieve unit variance
        Y_new = np.diag(1/np.std(Y,1)) @ Y # each row of Y is scaled by the inverse of its standard deviation
    elif name == 'PAYSIM':
        stds = np.std(Y, axis=1)
        safe_stds = np.clip(stds, 1e-8, None)  # don't allow std < 1e-8
        Y_new = np.diag(1 / safe_stds) @ Y

    return Y_new


def adjaceny_matrix(Y_norm, name, dimension):

    if name == 'AMLSIM':
        transactions = pd.read_csv(f'datasets/AMLSim/{dimension} users/transactions.csv')
        orig = "SENDER_ACCOUNT_ID"
        dest = "RECEIVER_ACCOUNT_ID"
        amnt = "TX_AMOUNT"

    elif name == 'PAYSIM':
        transactions = pd.read_csv(f'datasets/paysim/{dimension} users/rawLog.csv')
        orig = "nameOrig"
        dest = "nameDest"
        amnt = "amount"      

    elif name == 'LIBRA':
        transactions = pd.read_csv(f'datasets/libra/realdata/libra_380K.csv')
        orig = "id_source"
        dest = "id_destination"
        amnt = "cum_amount"

    users_list = []
    for _, row in transactions.iterrows():
        if row[orig] not in users_list:
            users_list.append(row[orig])
        if row[dest] not in users_list:
            users_list.append(row[dest])

    id_to_int = {user_id: idx for idx, user_id in enumerate(users_list)} 

    rows = len(Y_norm)

    matrix = np.zeros((rows, rows))

    for _, row in transactions.iterrows():
        if name == 'AMLSIM' or name == 'LIBRA':
            orig_acct = int(row[orig])
            bene_acct = int(row[dest])
            amount = float(row[amnt])
        else:
            orig_acct = int(id_to_int[row[orig]])
            bene_acct = int(id_to_int[row[dest]])
            amount = float(row[amnt])

        matrix[orig_acct][bene_acct] += amount

    mask = (matrix == 0) # cover all zeros

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap='YlGnBu', fmt=".2f", cbar=True, mask=mask,
                linewidths=0.5, linecolor='white')

    plt.xlabel("Receiver Account", fontsize=22)
    plt.ylabel("Sender Account", fontsize=22)
    plt.show()

    return matrix


def squic_fit_computation(Y_norm, name, dimension, adjaceny_matrix, printMatrix=False):
    with open('lambda_values.json') as f:
            lambda_data = json.load(f)
        
    lambdas = lambda_data[name][dimension]["norm"]

    ROWS = len(Y_norm)

    W_matrices = {}

    data_nnz = []
    data_nnzr = []
    data_time = []
    data_sym = []

    for rho in lambdas:
        W_matrices[rho], end_time = squic_fit_matrix(Y=Y_norm, l=rho, matrix=adjaceny_matrix)
        end_time = round(end_time, 2)
        print(f"required time: {end_time}")

        nnz, nnz_r = nnz_fit(W_matrices[rho], ROWS)
        print(f"nnz = {nnz} per rows = {nnz_r}")

        if printMatrix:
            sparsity_pattern(W_matrices[rho])

        if is_symmetric(W_matrices[rho]):
            print(f"✅ Matrix is symmetric per rho {rho}")
            data_sym.append("Yes")
        else:
            print(f"❌ Matrix is not symmetric per rho {rho}")
            data_sym.append("No")

        data_nnz.append(nnz)
        data_nnzr.append(nnz_r)
        data_time.append(end_time)

    table_fit_norm = [
            ["NNZ"] + data_nnz,
            ["NNZ/Row"] + data_nnzr,
            ["Time (s)"] + data_time,
            ["Symmetric"] + data_sym
    ]

    return W_matrices, table_fit_norm


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


def clustering(W_matrices):
    dict_cluster = {
        "louvain" : {},
        "spectral" : {},
        "dbscan" : {}
    }

    for rho, X in W_matrices.items():
        # Ensure all off-diagonal entries are positive
        X = np.abs(X) 

        # Ensure diagonal entries are zero
        np.fill_diagonal(X, 0)

        # Create a graph from matrix X
        G = nx.from_numpy_array(X)

        # Louvain
        partition = community.louvain_communities(G)
        # print(f"Number of cluster for rho {rho} is {len(partition)}")
        dict_cluster['louvain'][rho] = partition


        # DBSCAN with multiple params
        best_score = float('inf')
        best_params = None
        best_labels = None

        eps_range = np.linspace(0.1, 2.0, 20)
        min_samples_range = range(3, 10)

        eps_range = np.linspace(0.1, 2.0, 20)
        min_samples_range = range(3, 10)

        for eps in tqdm(eps_range):
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(np.asarray(X))
                
                # Ignore if all points are noise (-1) or single cluster
                if len(set(labels)) <= 1 or (set(labels) == {-1}):
                    continue

                partition = labels_to_partition(labels)

                diff = abs(len(partition) - len(dict_cluster['louvain'][rho]))
                if diff < best_score:
                    best_score = diff
                    best_params = (eps, min_samples)

        if best_params is not None:
            print(f"Best params: eps={best_params[0]}, min_samples={best_params[1]}")
            print(f"Best difference in number of clusters: {best_score:.4f}")
        else:
            print("No suitable DBSCAN parameters found!")
        print(f"Best difference in number of clusters: {best_score:.4f}")

        # DBSCAN
        if best_params is not None:
            dbscan = DBSCAN(eps=best_params[0], min_samples=best_params[1])
        else:
            dbscan = DBSCAN()
        labels_dbscan = dbscan.fit_predict(np.asarray(X))

        # Convert labels to list of sets
        partition_dbscan = labels_to_partition(labels_dbscan)
        dict_cluster['dbscan'][rho] = partition_dbscan


        # Spectral Clustering

        # Try with different n cluster and evaluate w/ siluette score
        # Define a range of number of clusters
        # cluster_range = range(2, 11)
        # sil_scores = []
        # cluster_labels = {}

        # for n in cluster_range:
        #     clustering = SpectralClustering(n_clusters=n, affinity='precomputed', assign_labels='cluster_qr')
        #     labels = clustering.fit_predict(np.asarray(X))

        #     # Compute the Silhouette Score
        #     score = silhouette_score(np.asarray(X), labels, metric='precomputed')
        #     sil_scores.append(score)
        #     cluster_labels[n] = labels

        # # Automatically choose the best number of clusters
        # best_n_clusters = cluster_range[np.argmax(sil_scores)]
        # labels_spectral = cluster_labels[best_n_clusters]

        # Use n_cluster = len(partitionin louvain)
        clustering = SpectralClustering(n_clusters=len(dict_cluster['louvain'][rho]), affinity='precomputed', assign_labels='cluster_qr')
        labels_spectral = clustering.fit_predict(np.asarray(X))

        # Convert labels to list of sets
        partition_spectral = labels_to_partition(labels_spectral)

        dict_cluster['spectral'][rho] = partition_spectral
    
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
            np.fill_diagonal(X, 0)

            # Modularity
            G = nx.from_numpy_array(X)

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


def visualize_metrics(metrics):
    # for every rho print RCut, NCut, Modularity and NCC
    for l, results in metrics.items():
        print(f"For rho = {l}")
        for method, res in results.items():
            print(f"    {method} : {res}")
        print("\n")
    return


def extract_results(metrics):
    # Initialize the rho_values list and dictionaries for each algorithm and metric
    rho_values = list(metrics.keys())
    
    # Initialize empty lists for each algorithm and metric
    lists_dict = {
        'rho_values': rho_values,
        'louvain_ncut': [],
        'dbscan_ncut': [],
        'spectral_ncut': [],
        'louvain_rcut': [],
        'dbscan_rcut': [],
        'spectral_rcut': [],
        'louvain_modularity': [],
        'dbscan_modularity': [],
        'spectral_modularity': [],
        'louvain_CC': [],
        'dbscan_CC': [],
        'spectral_CC': [],
        'louvain_nCluster': [],
        'dbscan_nCluster': [],
        'spectral_nCluster': []
    }
    
    # Extract values for each rho value
    for rho in rho_values:
        for algorithm in ['louvain', 'dbscan', 'spectral']:
            if algorithm in metrics[rho]:
                result = metrics[rho][algorithm]
                for metric in ['ncut', 'rcut', 'modularity', 'CC', 'nCluster']:
                    if metric in result:
                        lists_dict[f'{algorithm}_{metric}'].append(result[metric])
                    else:
                        lists_dict[f'{algorithm}_{metric}'].append(None)  # Handle missing metrics
            else:
                # Handle case where an algorithm is missing for a rho value
                for metric in ['ncut', 'rcut', 'modularity', 'CC', 'nCluster']:
                    lists_dict[f'{algorithm}_{metric}'].append(None)
    
    return lists_dict
    

def plot_results(metrics, ylabel, results, methods=['louvain', 'spectral', 'dbscan']):
    # Create the plot
    plt.figure(figsize=(6, 6))

    colors = ['steelblue', 'gold', 'indianred']
    dot = ['o-', 's-', '^-']

    if methods == ['louvain', 'spectral']:
        for i, method in enumerate(methods): 
            plt.plot(results['rho_values'], results[f'{method}_{metrics}'], dot[i], label=f'{method}', color=colors[i])
        
        plt.yscale('log')
    
    else:
        plt.plot(results['rho_values'], results[f'louvain_{metrics}'], 'o-', label='Louvain', color=colors[0])
        plt.plot(results['rho_values'], results[f'spectral_{metrics}'], 's-', label='Spectral', color=colors[1])
        plt.plot(results['rho_values'], results[f'dbscan_{metrics}'], '^-', label='DBSCAN', color=colors[2])

    plt.xlabel('Reg. Parameter (λ)', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.xscale('log')

    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.legend()
    plt.show()


def visualize_graph(W_matrices, dict_partition, account_prop, ds_name, ds_dimension):
    # colors show the clusters
    # intensity of color shows amount of money within this account (node weight)

    widget_dict = {
        rho: {
            'louvain' : {},
            'dbscan' : {},
            'spectral' : {}
        } for rho in W_matrices.keys()
    }
    
    for method, clustering in dict_partition.items():
        
        for l, X in W_matrices.items():
            # Create a graph from matrix X
            G = nx.from_numpy_array(X)

            # Extract clustering associated to matrix X with l=rho and method 
            partition = clustering[l]
            
            # Convert partition list of sets to a node-to-community mapping
            community_mapping = {}
            for community_id, community_set in enumerate(partition):
                for node in community_set:
                    community_mapping[node] = str(community_id)

            # Sets node attributes from a given value or dictionary of values.
            nx.set_node_attributes(G, community_mapping, 'community')

            fraud_mapping = {}
            for node in G.nodes():
                # Check if the node exists in account_prop
                if node in account_prop:
                    # print(f"node : {node} in account_prop -> fraud : {account_prop[node]['fraud']}")
                    fraud_mapping[node] = account_prop[node]['fraud']
                else:
                    # Default value if node not found in account_prop
                    fraud_mapping[node] = False
            
            # Set fraud attributes for all nodes
            nx.set_node_attributes(G, fraud_mapping, 'fraud')

            # Nodes: build dataframe from node attributes
            nodes_data = []

            fraudolent = []

            for node, data in G.nodes(data=True):
                is_fraud = data.get('fraud', False)

                # Create conditional label - only show label ID if fraudulent
                label = str(node) if is_fraud else ""

                if is_fraud:
                    fraudolent.append(str(node))

                nodes_data.append({
                    'id': node,
                    # 'label': str(node),
                    'label' : label,
                    'community': data.get('community'),
                    'color': data.get('color'),
                    'degree': G.degree[node],
                    'fraud' : is_fraud,
                })
            
            points = pd.DataFrame(nodes_data)

            points['community'] = points['community'].astype('category')

            # Edges: extract links with weights
            links_data = []
            for u, v, d in G.edges(data=True):
                links_data.append({
                    'source': u,
                    'target': v,
                    'weight': d.get('weight', 1.0)
                })
            links = pd.DataFrame(links_data)
            links['weight'] = links['weight'].abs()  # only positive thickness
            links['weight'] = links['weight'] * 10  # only positive thickness

            color_list = [
                ["0", "#4682B4"],  # Steelblue
                ["1", "#FF9999"],  # Red/Pink
                ["2", "#4169E1"],  # Royal Blue
                ["3", "#FFD700"],  # Yellow
                ["4", "#FFCC99"],  # Orange
                ["5", "#CCFFE5"],  # Light blue
                ["6", "#FFFF99"],  # Yellow
                ["7", "#CCFF99"],  # Lime
                ["8", "#CCFFFF"],  # Cyan
                ["9", "#CCCCFF"],  # Lilla
                ["10", "#E9967A"],  # Dark Salmon
                ["11", "#E5CCFF"],  # Dark Lilla
                ["12", "#DC143C"], # Crimson Red
                ["13", "#FFFFFF"], # White
                ["14", "#F0FFF0"],  # Honeydew,
            ]

            widget = cosmo(
                points=points,
                links=links,
                point_id_by='id', # id
                link_source_by='source',
                link_target_by='target',
                link_width_by='weight', # width of an edge given by weight = value in X between i and j
                # link_color_by='community',
                point_color_by='community',
                point_label_by='fraud',
                # point_label_by='id',
                point_size_by='degree', #'degree
                point_color_strategy='map',
                point_color_by_map=color_list,
                # background_color='#FFFFFF',
                show_labels_for=fraudolent
            )

            widget_dict[l][method] = widget
    
    return widget_dict
