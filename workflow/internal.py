from workflow.preprocess import *
from workflow.SQUIC_functions import *

import matplotlib.pyplot as plt
import matplotlib.colors as plt_color
import seaborn as sns
import json
import numpy as np
import networkx as nx
from networkx.algorithms import community
from cosmograph import cosmo


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
        df = AMLSim_preprocessing(dimension)
    elif name == 'PAYSIM':
        df = PaySim_preprocessing(dimension)
    elif name == 'LIBRE':
        # future implementation
        df = None
        print("LIBRE dataset support not yet implemented.")
    else:
        df = None

    return df, name, dimension


def extract_timeseries(df):
    # Plot the balance evolution for all users (columns) as separate lines
    # plt.figure(figsize=(15,10), dpi= 300)
    plt.figure(figsize=(15,10))

    # Plot each column (account balance) as a line
    for user in df.columns:
        # color = 'mediumseagreen' if user.startswith('C') else 'hotpink'

        plt.plot(df.index, df[user], label=f"User {user}", alpha=0.6)

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
        transactions = pd.read_csv(f'datasets/AMLSim/{dimension}/transactions.csv')
        orig = "SENDER_ACCOUNT_ID"
        dest = "RECEIVER_ACCOUNT_ID"
        amnt = "TX_AMOUNT"

    elif name == 'PAYSIM':
        transactions = pd.read_csv(f'datasets/paysim/{dimension}/rawLog.csv')
        orig = "nameOrig"
        dest = "nameDest"
        amnt = "amount"

        users_list = []

        for _, row in transactions.iterrows():
            if row[orig] not in users_list:
                users_list.append(row[orig])
            if row[dest] not in users_list:
                users_list.append(row[dest])

        id_to_int = {user_id: idx for idx, user_id in enumerate(users_list)}       

    elif name == 'LIBRA':
        transactions = pd.read_csv(f'datasets/libra/realdata/libra_380K.csv')
        orig = "id_source"
        dest = "id_destination"
        amnt = "cum_amount"

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

    plt.xlabel("Receiver Account")
    plt.ylabel("Sender Account")
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


def clustering(W_matrices):
    dict_cluster = {}

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

        dict_cluster[rho] = partition
    
    return dict_cluster
    

def internal_metrics(dict_cluster, W_matrices):
    int_metrics = {rho: {} for rho in W_matrices.keys()}

    for l, X in W_matrices.items():

        partition = dict_cluster[l]
        node_to_community = {}
        for idx, comm in enumerate(partition):
            for node in comm:
                node_to_community[node] = idx

        # labels = [node_to_community[n] for n in range(len(node_to_community))] # print the label of where every node is

        # Ensure all off-diagonal entries are positive
        X = np.abs(X) 

        # Ensure diagonal entries are zero
        np.fill_diagonal(X, 0)

        # Modularity
        G = nx.from_numpy_array(X)

        # unique_labels = np.unique(labels)

        n_cut = 0
        r_cut = 0
        
        # for cluster in unique_labels:
        #     mask = (labels == cluster)
        #     not_mask = ~mask
        #     cut = X[mask][:, not_mask].sum()
        #     vol = X[mask].sum()
        #     assoc = X[mask][:, mask].sum()
            
        #     n_cut += cut / (vol + 1e-10)  # Avoid division by zero
        #     r_cut += cut / (mask.sum() + 1e-10)  # Normalize by cluster size

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

        int_metrics[l]["ncut"] = float(round(n_cut, 2))
        int_metrics[l]["rcut"] = float(round(r_cut, 2))

        modularity = community.modularity(G, dict_cluster[l])
        int_metrics[l]['modularity'] = float(round(modularity, 2))
        
        # # Strongly Connected Components
        # if not G.is_directed():
        #     G_dir = G.to_directed()
        # else:
        #     G_dir = G

        # Connected Components
        int_metrics[l]['CC'] = nx.number_connected_components(G)

    return int_metrics


def visualize_metrics(metrics):
    # for every rho print RCut, NCut, Modularity and NCC
    for l, results in metrics.items():
        print(f"For rho {l} : {results}")
    return
    

def visualize_graph(W_matrices, dict_partition, ds_name, ds_dimension):
    # colors show the clusters
    # intensity of color shows amount of money within this account (node weight)

    widget_dict = {}

    for l, X in W_matrices.items():
        # Create a graph from matrix X
        G = nx.from_numpy_array(X)

        # Extract clustering associated to matrix X with l=rho
        partition = dict_partition[l]
        
        # Convert partition list of sets to a node-to-community mapping
        community_mapping = {}
        for community_id, community_set in enumerate(partition):
            for node in community_set:
                community_mapping[node] = str(community_id)

        # Sets node attributes from a given value or dictionary of values.
        nx.set_node_attributes(G, community_mapping, 'community')

        # Nodes: build dataframe from node attributes
        nodes_data = []
        for node, data in G.nodes(data=True):
            nodes_data.append({
                'id': node,
                'label': str(node),
                'community': data.get('community'),
                'color': data.get('color'),
                'degree': G.degree[node] # how many non zero entries there are for a node X
            })
        points = pd.DataFrame(nodes_data)

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

        widget = cosmo(
            points=points,
            links=links,
            point_id_by='id',
            link_source_by='source',
            link_target_by='target',
            link_width_by='weight', # width of an edge given by weight = value in X between i and j
            link_color_by='community',
            point_color_by='community',
            point_label_by='label', # or community
            point_size_by='degree'        
        )

        widget_dict[l] = widget
    
        # Show with NX

        # Map community labels to integers for coloring
        # unique_communities = list(points['community'].unique())
        # community_to_int = {community: idx for idx, community in enumerate(unique_communities)}
        # node_colors = points['community'].map(community_to_int)

        # # Normalize colors for colormap
        # cmap = cm.get_cmap('tab10', len(unique_communities))

        # # Define node sizes (scaled degree)
        # node_sizes = points['degree'] * 100  # or change the multiplier for bigger/smaller nodes

        # # Generate layout
        # pos = nx.spring_layout(G, seed=42)  # spring layout works well for general-purpose

        # # Create the figure
        # plt.figure(figsize=(12, 9))

        # # Draw nodes with color and size
        # nx.draw_networkx_nodes(
        #     G, pos,
        #     node_color=node_colors,
        #     node_size=node_sizes,
        #     cmap=cmap,
        #     alpha=0.9
        # )

        # # Draw edges
        # nx.draw_networkx_edges(G, pos, alpha=0.4)

        # # Draw labels (optional)
        # nx.draw_networkx_labels(G, pos, font_size=10)

        # # Add legend
        # for community, idx in community_to_int.items():
        #     plt.scatter([], [], c=[cmap(idx)], label=str(community), s=100)
        # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title="Community")

        # plt.title(f"Graph for rho = {l}", fontsize=14)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

    return widget_dict
