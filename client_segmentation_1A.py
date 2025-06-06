from workflow.internal import load_dataset, knn_graph, normalization, visualize_metrics, extract_timeseries
from workflow.SQUIC_functions import squic_fit_matrix_computation, squic_fit_computation
from workflow.clustering_functions import clustering_2_communities, modularity_fscore, ARI_fscore, plot_ARI_f1, labels_to_partition, partition_to_labels
import matplotlib.pyplot as plt
import networkx as nx
import json
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os


def ask_yes_no(prompt, default=None):
    """
    Ask the user a yes/no question with optional default.
    """
    while True:
        answer = input(f"{prompt} (Y/N) ").strip().upper()
        if not answer and default:
            return default
        if answer in ['Y', 'N']:
            return answer
        print("Please enter 'Y' or 'N'.")


def ask_input(prompt, default=None):
    """
    Ask the user for input with optional default.
    """
    answer = input(f"{prompt} ").strip()
    if not answer and default is not None:
        return default
    return answer


if __name__ == '__main__':
    user_input = ask_yes_no("\nDo you want to load data?")

    if user_input == 'Y':
        dimension = ask_input("Which dimension? (100/1K/10K/100K/1M)").upper()
        path = 'paysim_data_saved'

        try:
            # Load pre-saved Y_norm
            chunk_size = 10000
            chunks = []

            for chunk in pd.read_csv(f'{path}/YNorm_{dimension}.csv', chunksize=chunk_size):
                chunks.append(chunk)

            Y_norm = pd.concat(chunks, ignore_index=True)

            # Load pre-saved knn_matrix
            knn_matrix = sparse.load_npz(f'{path}/knn_matrix_{dimension}.npz')
            
            # Load pre-saved account_properties
            with open(f'{path}/account_prop_{dimension}.json', 'r') as f:
                account_prop = json.load(f)

            account_prop = {int(k): v for k, v in account_prop.items()}
            
            name = 'PAYSIM'

        except Exception as e:
            # If data are not present
            print(f"Error loading data: {e}")
            exit(1)
        
    else: # Normal run
        Y, name, dimension, account_prop, _ = load_dataset()

        Y_norm = normalization(Y, name)

        if dimension == '100':
            n_neighbors = 10
        elif dimension == '1K':
            n_neighbors = 7
        elif dimension == '10K':
            n_neighbors = 3
        else:
            n_neighbors = 2

        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', n_jobs=-1)
        nbrs.fit(Y_norm)
        knn_matrix = nbrs.kneighbors_graph(Y_norm, mode='connectivity')
        
        # Ask user if want to save the data for next runs
        if ask_yes_no("Do you want to cache this data?") == 'Y':
            path = 'paysim_data_saved'

            # if path does not exists, create it
            os.makedirs(path, exist_ok=True)
            
            # Save Y_norm as CSV
            pd.DataFrame(Y_norm).to_csv(f'{path}/YNorm_{dimension}.csv', index=False)
            
            # Save account_prop as json
            with open(f'{path}/account_prop_{dimension}.json', 'w') as f:
                json.dump(account_prop, f)
            
            # Save knn_matrix as npz
            sparse.save_npz(f'{path}/knn_matrix_{dimension}.npz', knn_matrix)

    
    # Print knn matrix
    plt.figure(figsize=(7, 7))
    plt.spy(knn_matrix, markersize=5)
    plt.ylabel("Users", fontsize=18)
    plt.show()

    results_squic = {}

    # Run SQUIC_fit
    if ask_yes_no("SQUIC-Fit with bias or no?") == 'Y':
        if ask_yes_no("Do you want to visualize the results of SQUIC-Fit?") == 'Y':
            printMatrix = True
        else:
            printMatrix = False
        results_squic['squic-fit-matrix'] = squic_fit_matrix_computation(Y_norm, name, dimension, knn_matrix, printMatrix)
    
    else:
        if ask_yes_no("Do you want to visualize the results of SQUIC-Fit?") == 'Y':
            printMatrix = True
        else:
            printMatrix = False
        results_squic['squic-fit-matrix'] = squic_fit_computation(Y_norm, name, dimension, printMatrix)

    # Results
    dict_cluster = clustering_2_communities(results_squic, method='implemented')

    metrics = ARI_fscore(dict_cluster, results_squic, account_prop)

    for _, data in metrics.items():
        for rho, results in data.items():
            print(f"for rho = {rho} : {results}")

    plot_ARI_f1(metrics)
    