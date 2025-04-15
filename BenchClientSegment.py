from preprocessing.preprocess import *
import matplotlib.pyplot as plt
import json
import numpy as np
from squic_folder.SQUIC_functions import *
import networkx as nx
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering


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


def squic_computation(Y_norm, name, dimension, printMatrix=False):
    with open('lambda_values.json') as f:
            lambda_data = json.load(f)
        
    lambdas = lambda_data[name][dimension]["norm"]

    ROWS = len(Y_norm)

    fit_norm_dict = {}

    data_nnz = []
    data_nnzr = []
    data_time = []
    data_sym = []

    for rho in lambdas:
        fit_norm_dict[rho], end_time = squic_fit(Y_norm, lambda_val=rho, eta=rho * 0.1)
        end_time = round(end_time, 2)
        print(f"required time: {end_time}")

        nnz, nnz_r = nnz_fit(fit_norm_dict[rho], ROWS)
        print(f"nnz = {nnz} per rows = {nnz_r}")

        if printMatrix:
            sparsity_pattern(fit_norm_dict[rho])

        if is_symmetric(fit_norm_dict[rho]):
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

    return fit_norm_dict, table_fit_norm, lambdas


def clustering(dict_results, lambdas):
    dict_labels = {}

    for rho in lambdas:
        X = dict_results[rho]

        # Ensure all off-diagonal entries are positive
        X = np.abs(X) 

        # Ensure diagonal entries are zero
        np.fill_diagonal(X, 0)

        # Create a graph from matrix X
        G = nx.from_numpy_array(X)

        # Define a range of number of cluster
        cluster_range = range(2, 11)
        sil_scores = []
        cluster_labels = {}

        for n in cluster_range:
            # Perform SpectralClustering
            clustering = SpectralClustering(n_clusters=n, affinity='precomputed', assign_labels='cluster_qr')
            labels = clustering.fit_predict(np.asarray(X))
            # labels = clustering.fit_predict(G)

            
            # Compute the Silhouette Score
            score = silhouette_score(np.asarray(X), labels, metric='precomputed')
            # score = silhouette_score(G, labels, metric='precomputed')
            sil_scores.append(score)
            cluster_labels[n] = labels

        # Plot Silhouette Score for each number of clusters
        plt.plot(cluster_range, sil_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score Method')
        plt.show()

        # Automatically choose the best number of clusters
        best_n_clusters = cluster_range[np.argmax(sil_scores)]
        labels = cluster_labels[best_n_clusters]

        print(f"Best number of clusters for rho={rho}: {best_n_clusters}")

        dict_labels[rho] = labels
    
    return dict_labels
    

def internal_metrics(dict_labels, adjaceny_matrices, lambdas):
    for rho in lambdas:
        labels = dict_labels[rho]

        # Normalized Cut and Ratio Cut
        X = adjaceny_matrices[rho]
        n_cut = 0

        unique_labels = np.unique(labels)
        r_cut = 0
        
        for cluster in unique_labels:
            mask = (labels == cluster)
            not_mask = ~mask
            cut = X[mask][:, not_mask].sum()
            vol = X[mask].sum()
            assoc = X[mask][:, mask].sum()
            
            n_cut += cut / (vol + 1e-10)  # Avoid division by zero
            r_cut += cut
        
        print(n_cut)
        print(r_cut / len(unique_labels))
        
        # Ratio Cut
        
        # Modularity
        
        # Strongly Connected Components
    

def create_graph(X):
    pass


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    df, name, dimension = load_dataset()

    # Extract time series
    extract_timeseries(df)

    # Normalise time series
    Y_norm = normalization(df, name)

    # Run SQUIC_fit
    adjaceny_matrices, table_results, lambdas = squic_computation(Y_norm, name, dimension)
    
    # Use the extracted W for clustering
    dict_labels = clustering(adjaceny_matrices, lambdas)

    # Report internal metrics on the clustering
    internal_metrics(dict_labels, adjaceny_matrices, lambdas)

    # Visualise with cosmograph
    # create_graph()