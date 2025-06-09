from workflow.internal import load_dataset, normalization, extract_timeseries
from workflow.SQUIC_functions import squic_fit_matrix_computation, squic_fit_computation
from workflow.clustering_functions import clustering_2_communities, ARI_fscore, plot_ARI_f1
import matplotlib.pyplot as plt
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
    '''
    This script allows you to **load previously cached data** to speed up subsequent runs. 
    Specifically, it can load:
    - `Y_normalized` (the normalized dataset)
    - `knn_matrix` (the nearest neighbors matrix)
    - `account_prop` (the account properties)

    These files are expected to be stored in the `paysim_data_saved` directory, named with the 
    corresponding dataset dimension.

    Note:
    - The data must have been saved from a **previous successful run** using the same dimension.
    - On the **first run** for a given dimension, you must generate and save this data by answering 
      "Y" when prompted to cache it.
    '''
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

        extract_timeseries(Y, name)

        Y_norm = normalization(Y, name)

        n_neigs = {
            '100' : 10,
            '1K' : 8,
            '10K' : 4,
            '100K' : 2,
            '1M' : 2
        }

        nbrs = NearestNeighbors(n_neighbors=n_neigs[dimension], metric='euclidean', n_jobs=-1)
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

    results_squic = {}

    # Run SQUIC_fit
    if ask_yes_no("SQUIC-Fit with bias or no?") == 'Y':
        
        # Print knn matrix
        plt.figure(figsize=(7, 7))
        plt.spy(knn_matrix, markersize=5)
        plt.ylabel("Users", fontsize=18)
        plt.savefig(f'images/{dimension}/knn_matrix')
        plt.show()

        if ask_yes_no("Do you want to visualize the results of SQUIC-Fit?") == 'Y':
            printMatrix = True
        else:
            printMatrix = False

        if ask_yes_no("Do you want to save the results of SQUIC-Fit?") == 'Y':
            save = True
        else:
            save = False
        
        squic_method = 'squic-fit-matrix'
        results_squic[squic_method] = squic_fit_matrix_computation(Y_norm, name, dimension, knn_matrix, printMatrix, save)
    
    else:
        if ask_yes_no("Do you want to visualize the results of SQUIC-Fit?") == 'Y':
            printMatrix = True
        else:
            printMatrix = False

        if ask_yes_no("Do you want to save the results of SQUIC-Fit?") == 'Y':
            save = True
        else:
            save = False

        squic_method = 'squic-fit'
        results_squic[squic_method] = squic_fit_computation(Y_norm, name, dimension, printMatrix, save)

    dict_method = {
        '100' : 'scikit-learn',
        '1K' : 'implemented',
        '10K' : 'implemented',
        '100K' : 'implemented',
        '1M' : 'implemented'
    }

    # Results
    dict_cluster = clustering_2_communities(results_squic, dimension, squic_method, method=dict_method[dimension])

    metrics = ARI_fscore(dict_cluster, results_squic, account_prop)

    for _, data in metrics.items():
        for rho, results in data.items():
            print(f"for rho = {rho} : {results}")

    plot_ARI_f1(metrics, squic_method, dimension, save)
    