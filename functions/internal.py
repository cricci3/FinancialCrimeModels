from functions.preprocess import *
from functions.SQUIC_functions import *

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for custom legend
import seaborn as sns

import json
import numpy as np

from sklearn.neighbors import NearestNeighbors
import networkx as nx
from cosmograph import cosmo


def parse_input(user_input, name_dataset):
    """Parse and validate the dataset input, which can be 'NAME_DIMENSION' or just 'NAME' (e.g., LIBRA)."""
    user_input = user_input.strip().upper()
    parts = user_input.split("_")

    valid_names_with_dimensions = {name_dataset}
    valid_names_without_dimensions = {"LIBRA"}
    valid_dimensions = {"100", "1K", "10K", "100K", "1M"}

    # if len(parts) == 1:
    #     name = parts[0]
    #     if name not in valid_names_without_dimensions:
    #         raise ValueError(f"Invalid dataset name '{name}'. Valid options are: "
    #                          f"{', '.join(valid_names_with_dimensions | valid_names_without_dimensions)}")
    #     return name, None

    if len(parts) == 2:
        name, dimension = parts
        if name not in valid_names_with_dimensions:
            raise ValueError(f"Invalid dataset name '{name}'. Valid options for dimensioned datasets are: "
                             f"{', '.join(valid_names_with_dimensions)}")
        if dimension not in valid_dimensions:
            raise ValueError(f"Invalid dimension '{dimension}'. Valid options are: {', '.join(valid_dimensions)}")
        return name, dimension

    else:
        # raise ValueError("Input must be in the format NAME_DIMENSION (e.g., PAYSIM_10K) or just NAME (e.g., LIBRA)")
        raise ValueError(f"Input must be in the format NAME_DIMENSION (e.g., {name_dataset}_10K)")


def load_dataset_1A():
    """Prompt user input and load the corresponding dataset."""
    while True:
        user_input = input("Insert dataset name in the following format NAME_DIMENSION (e.g., PAYSIM_10K): ")

        try:
            name, dimension = parse_input(user_input, "PAYSIM")
            break
        except ValueError as e:
            print(f"Error: {e}")
            continue

    if name == 'PAYSIM':
        # account prop for paysim is different, contains "class" also the type of user: B, C, M
        df, account_prop = PaySim_preprocessing(dimension)
    else:
        df = None

    return df, name, dimension, account_prop


def load_dataset_1B():
    """Prompt user input and load the corresponding dataset."""
    while True:
        user_input = input("Insert dataset name in the following format NAME_DIMENSION (e.g., AMLSIM_10K): ")

        try:
            name, dimension = parse_input(user_input, "AMLSIM")
            break
        except ValueError as e:
            print(f"Error: {e}")
            continue

    if name == 'AMLSIM':
        df, account_prop = AMLSim_preprocessing(dimension)
    else:
        df = None

    return df, name, dimension, account_prop


def extract_timeseries(df, name, dimension=None, type_df=None):
    # Plot the balance evolution for all users (columns) as separate lines
    # plt.figure(figsize=(15,10), dpi= 300)
    plt.figure(figsize=(7,7))

    if type_df == 'norm':
        df = df.T

        if name == 'AMLSIM':
            n_days, n_users = df.shape
            # Create labels
            user_labels = [f"User_{i}" for i in range(n_users)]
            day_labels = list(range(n_days))

            # Create the DataFrame
            df = pd.DataFrame(df, index=day_labels, columns=user_labels)

    if name == 'PAYSIM' and type_df != 'norm':
        colors = {'Clients': 'mediumseagreen', 
              'Bank': 'crimson', 
              'Merchants': 'darkturquoise'}
        
        bank_dimension = ['100', '1K', '10K']

        # Plot each column (account balance) as a line
        for user in df.columns:
            if user.startswith('C'):
                color = colors['Clients']
                plt.plot(df.index, df[user], color=color, alpha=0.6)
            elif user.startswith('B'):
                color = colors['Bank']
                if dimension in bank_dimension:
                    plt.plot(df.index, df[user], color=color, alpha=0.6)
            elif user.startswith('M'):
                color = colors['Merchants']
                plt.plot(df.index, df[user], color=color, alpha=0.6)
            else:
                plt.plot(df.index, df[user], alpha=0.6)
            # plt.plot(df.index, df[user], color=color, alpha=0.6)
        
        # plt.legend(colors, ['Clients', 'Bank', 'Merchands'])
    
        if dimension in bank_dimension:
            legend_elements = [
                Line2D([0], [0], color=colors.get('Clients'), lw=4, label='Clients'),
                Line2D([0], [0], color=colors.get('Bank'), lw=4, label='Bank'),
                Line2D([0], [0], color=colors.get('Merchants'), lw=4, label='Merchants')
            ]
        else:
            legend_elements = [
                Line2D([0], [0], color=colors.get('Clients'), lw=4, label='Clients'),
                Line2D([0], [0], color=colors.get('Merchants'), lw=4, label='Merchants')
            ]

        if type_df != 'norm':
            plt.legend(handles=legend_elements, fontsize=16)
    
    else:
        # Plot each column (account balance) as a line
        for user in df.columns:
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


def prepare_timeseries(df):
    # Check if float
    Y = df.astype(float)

    # --- First-differencing to get stationary data
    # Compute the difference across time (columns, axis=1)
    Y_changes = Y.diff(axis=1)

    # First column will be NaN
    Y_changes = Y_changes.dropna(axis=1)

    # Standardization of the differenced data
    # Get the numpy array
    Y_matrix = Y_changes.values

    # Compute std dev per row of the differenced data
    row_std = np.std(Y_matrix, axis=1, keepdims=True)

    # No division by zero
    row_std[row_std == 0] = 1.0

    # Normalize each row (time series of an account's *changes*)
    Y_prepared = Y_matrix / row_std

    # Convert back to a DataFrame with original users and updated days
    return pd.DataFrame(Y_prepared, index=Y_changes.index, columns=Y_changes.columns)


def normalization(df, name):
    if name == 'AMLSIM':
        # # df is user on the rows and days on the columns
        # Y = df
        # # Compute std dev per row
        # row_std = np.std(Y, axis=1, keepdims=True)
        # # Avoid division by zero -> when the balance of a user is constant
        # row_std[row_std == 0] = 1.0
        # # Normalize each row (timeseries of a user)
        # Y_new = Y / row_std
        # return Y_new
        
        Y = df
        stds = np.std(Y, axis=1)
        safe_stds = np.clip(stds, 1e-8, None)  # don't allow std < 1e-8
        Y_new = np.diag(1 / safe_stds) @ Y
        return Y_new
    
    elif name == 'PAYSIM':
        # Y = df.T.values  # Convert to (users, days)
        # Y = df.T
        Y = df

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


def visualize_graph_internal(W_matrices, squic_method, dict_partition, account_prop, name, two_comm=True):
    # colors show the clusters
    # intensity of color shows amount of money within this account (node weight)

    widget_dict = {
        rho: {
            'louvain' : {},
            'dbscan' : {},
            'spectral' : {}
        } for rho in W_matrices[squic_method].keys()
    }
    
    for rho, partition in dict_partition[squic_method].items():
        
        for rho, X in W_matrices[squic_method].items():
            # Create a graph from matrix X
            G = nx.from_numpy_array(X)

            # Extract clustering associated to matrix X with l=rho and method 
            # partition = clustering[rho]
            
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
            # links['weight'] = links['weight'] * 10  # only positive thickness
            
            if two_comm == True:
                color_list = [
                    ["0", "#90ee90"],  # 'Clients': 'mediumseagreen'
                    ["1", "#00CED1"],  # 'Merchants': 'darkturquoise'
                ]
            else:
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

            # COSMOGRAPH docs: https://colab.research.google.com/drive/1Rt8rmmeMuWyFjEqae2DdJ3NYymtjC9cT#scrollTo=IZUK7ioL1xKr
            widget = cosmo(
                points=points,
                links=links,
                point_id_by='id', # id
                link_source_by='source',
                link_target_by='target',
                link_width_by='weight', # width of an edge given by weight = value in X between i and j
                link_color='#E3E3E3',
                link_greyout_opacity=0.1,
                # link_color_by='community',
                point_color_by='community',
                point_label_by='class', # or id or fraud 
                point_size_by='degree', #'degree
                point_color_strategy='map',
                point_color_by_map=color_list,
                background_color='#FFFFFF',
                show_labels_for=fraudolent
            )

            widget_dict[rho]['spectral'] = widget
    
    return widget_dict
