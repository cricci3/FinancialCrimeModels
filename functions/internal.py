from functions.preprocess import *
from functions.SQUIC_functions import *

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])
from matplotlib.lines import Line2D
import seaborn as sns

import json
import numpy as np

from sklearn.neighbors import NearestNeighbors
import networkx as nx
from cosmograph import cosmo


def parse_input(user_input):
    """Parse and validate the dataset input, which can be 'NAME_DIMENSION'."""
    user_input = user_input.strip().upper()
    parts = user_input.split("_")

    valid_names_with_dimensions = {"AMLSIM", "PAYSIM"}
    valid_dimensions = {"100", "1K", "10K", "100K"}

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
        raise ValueError(f"Input must be in the format NAME_DIMENSION (e.g., AMLSIM_10K)")


def load_dataset_paysim():
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


def load_dataset_amlsim():
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
    else:
        df = None

    return df, name, dimension, account_prop


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
