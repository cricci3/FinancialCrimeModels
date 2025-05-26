from workflow.preprocess import *
from workflow.SQUIC_functions import *

import matplotlib.pyplot as plt
import seaborn as sns

import json
import numpy as np

import networkx as nx
from cosmograph import cosmo


def parse_input(user_input):
    """Parse and validate the dataset input, which can be 'NAME_DIMENSION' or just 'NAME' (e.g., LIBRA)."""
    user_input = user_input.strip().upper()
    parts = user_input.split("_")

    valid_names_with_dimensions = {"AMLSIM", "PAYSIM"}
    valid_names_without_dimensions = {"LIBRA"}
    valid_dimensions = {"100", "1K", "10K", "100K", "1M"}

    if len(parts) == 1:
        name = parts[0]
        if name not in valid_names_without_dimensions:
            raise ValueError(f"Invalid dataset name '{name}'. Valid options are: "
                             f"{', '.join(valid_names_with_dimensions | valid_names_without_dimensions)}")
        return name, None

    elif len(parts) == 2:
        name, dimension = parts
        if name not in valid_names_with_dimensions:
            raise ValueError(f"Invalid dataset name '{name}'. Valid options for dimensioned datasets are: "
                             f"{', '.join(valid_names_with_dimensions)}")
        if dimension not in valid_dimensions:
            raise ValueError(f"Invalid dimension '{dimension}'. Valid options are: {', '.join(valid_dimensions)}")
        return name, dimension

    else:
        raise ValueError("Input must be in the format NAME_DIMENSION (e.g., AMLSIM_10K) or just NAME (e.g., LIBRA)")


def load_dataset():
    """Prompt user input and load the corresponding dataset."""
    while True:
        user_input = input("Insert dataset name in the following format NAME_DIMENSION (e.g., AMLSIM_10K) or LIBRA: ")

        try:
            name, dimension = parse_input(user_input)
            break
        except ValueError as e:
            print(f"Error: {e}")
            continue

    if name == 'AMLSIM':
        df, account_prop, trans_matrix = AMLSim_preprocessing(dimension)
    elif name == 'PAYSIM':
        # account prop for paysim is different, contains "class" also the type of user: B, C, M
        df, account_prop, trans_matrix = PaySim_preprocessing(dimension)
    elif name == 'LIBRA':
        # future implementation
        df, account_prop, trans_matrix = Libra_preprocessing()
    else:
        df = None

    return df, name, dimension, account_prop, trans_matrix


def extract_timeseries(df, name):
    # Plot the balance evolution for all users (columns) as separate lines
    # plt.figure(figsize=(15,10), dpi= 300)
    plt.figure(figsize=(7,7))

    if name == 'PAYSIM':
        # Plot each column (account balance) as a line
        for user in df.columns:
            if user.startswith('C'):
                color = 'mediumseagreen'
            elif user.startswith('B'):
                color = 'crimson'
            else:
                color = 'darkturquoise'
            plt.plot(df.index, df[user], color=color, label=f"User {user}", alpha=0.6)
    
    else:
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
    Y = df.T.values  # Convert to (users, days)

    if name == 'AMLSIM':
        # normalize the Y input data to achieve unit variance
        Y_new = np.diag(1/np.std(Y,1)) @ Y # each row of Y is scaled by the inverse of its standard deviation
        return Y_new
    elif name == 'PAYSIM':
        stds = np.std(Y, axis=1)
        safe_stds = np.clip(stds, 1e-8, None)  # don't allow std < 1e-8
        Y_new = np.diag(1 / safe_stds) @ Y
        return Y_new
    elif name == 'LIBRA':
        return Y


def print_transaction_matrix(matrix):
    mask = (matrix == 0) # cover all zeros

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap='YlGnBu', fmt=".2f", cbar=True, mask=mask,
                linewidths=0.5, linecolor='white')

    plt.xlabel("Receiver Account", fontsize=22)
    plt.ylabel("Sender Account", fontsize=22)
    plt.show()

    return


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


def visualize_graph_internal(W_matrices, dict_partition, account_prop, name):
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
            class_mapping = {}
            for node in G.nodes():
                if node in account_prop and isinstance(account_prop[node], dict):
                    if 'fraud' in account_prop[node]:
                        fraud_mapping[node] = account_prop[node]['fraud']
                    if name == 'PAYSIM' and 'class' in account_prop[node]:
                        class_mapping[node] = account_prop[node]['class']
                else:
                    # Default value if node not found in account_prop
                    fraud_mapping[node] = False
            
            # Set fraud attributes for all nodes
            nx.set_node_attributes(G, fraud_mapping, 'fraud')
            
            if name == 'PAYSIM':
                nx.set_node_attributes(G, class_mapping, 'class')

            # Nodes: build dataframe from node attributes
            nodes_data = []
            fraudolent = []

            for node, data in G.nodes(data=True):
                is_fraud = data.get('fraud', False)
                if name == 'PAYSIM':
                    user_class = data.get('class', False)

                # Create conditional label - only show label ID if fraudulent
                label = str(node) if is_fraud else ""

                if is_fraud:
                    fraudolent.append(str(node))

                if name != 'PAYSIM':
                    nodes_data.append({
                        'id': node,
                        # 'label': str(node),
                        'label' : label,
                        'community': data.get('community'),
                        'color': data.get('color'),
                        'degree': G.degree[node],
                        'fraud' : is_fraud,
                    })
                else:
                    nodes_data.append({
                    'id': node,
                    # 'label': str(node),
                    'label' : label,
                    'community': data.get('community'),
                    'color': data.get('color'),
                    'degree': G.degree[node],
                    'fraud' : is_fraud,
                    'class' : user_class
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
                point_label_by='class', # or id or fraud 
                point_size_by='degree', #'degree
                point_color_strategy='map',
                point_color_by_map=color_list,
                # background_color='#FFFFFF',
                show_labels_for=fraudolent
            )

            widget_dict[l][method] = widget
    
    return widget_dict
