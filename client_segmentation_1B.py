from functions.internal import load_dataset, normalization, extract_timeseries
from functions.SQUIC_functions import squic_fit_matrix_computation, squic_fit_computation, check_symmetric_sparse
from functions.clustering_functions import clustering_optimal_number, modularity_density, plot_PDens_Q
import matplotlib.pyplot as plt
import json
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import os


def show_df(Y):
    n_users, n_days = Y.shape

    # Create labels
    day_labels = [f"Day {i}" for i in range(n_days)]
    user_labels = list(range(n_users))

    # Create the DataFrame
    df = pd.DataFrame(Y, index=user_labels, columns=day_labels)

    print(df)


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
        dimension = ask_input("Which dimension (100/1K/10K/100K/1M)?").upper()
        path = 'amlsim_data_saved'

        try:
            # Load pre-saved Y_norm
            chunk_size = 10000
            chunks = []

            for chunk in pd.read_csv(f'{path}/YNorm_{dimension}.csv', chunksize=chunk_size):
                chunks.append(chunk)

            Y_norm = pd.concat(chunks, ignore_index=True)

            print(f"Head of Y_norm df:")
            print(Y_norm.head())
            print(f"\ndimension of YNorm loaded: {Y_norm.shape}")

            std_dev = round(np.mean(np.std(Y_norm, axis=1)), 2)

            print("Mean std per row (should be 1):", std_dev)
            print("Max:", np.max(Y_norm))
            print("Min:", np.min(Y_norm))

            # Load pre-saved knn_matrix
            knn_matrix = sparse.load_npz(f'{path}/knn_matrix_{dimension}.npz')

            print("Shape KNN:", knn_matrix.shape)
            print("nnz KNN:", knn_matrix.nnz)

            # Average number of non-zeros per row
            nnz_per_row = knn_matrix.nnz / knn_matrix.shape[0]
            print("nnz per row:", round(nnz_per_row, 2))

            print("Is symmetric:", check_symmetric_sparse(knn_matrix))

            # Print knn matrix
            # plt.figure(figsize=(7, 7))
            # plt.spy(knn_matrix, markersize=5)
            # plt.ylabel("Users", fontsize=18)
            # plt.title("KNN")
            # plt.show()
            
            # Load pre-saved account_properties
            with open(f'{path}/account_prop_{dimension}.json', 'r') as f:
                account_prop = json.load(f)

            account_prop = {int(k): v for k, v in account_prop.items()}
            
            name = 'AMLSIM'

        except Exception as e:
            # If data are not present
            print(f"Error loading data: {e}")
            exit(1)
        
    else: # Normal run
        Y, name, dimension, account_prop = load_dataset()

        extract_timeseries(Y, name)

        Y = Y.T.values  # Convert to (users, days)

        show_df(Y)

        # if ask_yes_no("Do you want to reorder data?") == 'Y':
        #     # ---- delta
        #     # Compute delta: last day - first day
        #     delta = Y[:, -1] - Y[:, 0]  # shape (n_users,)

        #     # Get sort order (descending: most positive trend first)
        #     sort_indices = np.argsort(-delta)

        #     # Reorder the rows of Y based on trend
        #     Y = Y[sort_indices]

        #     show_df(Y)

        Y_norm = normalization(Y, name)
        extract_timeseries(Y_norm, name, type_df='norm')

        show_df(Y_norm)

        # n_users, n_days = Y.shape

        # # Create labels (optional: you can customize these)
        # user_labels = [f"User_{i}" for i in range(n_users)]
        # day_labels = [f"{j}" for j in range(n_days)]

        # # Create the DataFrame
        # df = pd.DataFrame(Y, index=user_labels, columns=day_labels)

        # Y_norm2 = prepare_timeseries(df)

        # print(f"shape of Y_norm2 = {Y_norm2.shape}")

        # plt.figure(figsize=(7, 7))

        # for user in Y_norm2.index:
        #     plt.plot(Y_norm2.columns, Y_norm2.loc[user], label=user, alpha=0.6)

        # plt.xlabel("Days", fontsize=16)
        # plt.ylabel("Balance", fontsize=16)
        # plt.tight_layout()
        # plt.show()

        print(f"\ndimension of YNorm loaded: {Y_norm.shape}")

        std_dev = round(np.mean(np.std(Y_norm, axis=1)), 2)

        print("Mean std per row (should be 1):", std_dev)
        print("Max:", np.max(Y_norm))
        print("Min:", np.min(Y_norm))

        n_neigs = {
            '100' : 10,
            '1K' : 5,
            '10K' : 5,
            '100K' : 4,
            '1M' : 3
        }

        nbrs = NearestNeighbors(n_neighbors=n_neigs[dimension], metric='euclidean', n_jobs=-1)
        nbrs.fit(Y_norm)
        knn_matrix = nbrs.kneighbors_graph(Y_norm, mode='connectivity')

        # knn_matrix = knn_matrix.multiply(knn_matrix.T)

        print("Shape KNN:", knn_matrix.shape)
        print("nnz KNN:", knn_matrix.nnz)

        # Average number of non-zeros per row
        nnz_per_row = knn_matrix.nnz / knn_matrix.shape[0]
        print("nnz per row:", round(nnz_per_row, 2))
        print("Is symmetric:", check_symmetric_sparse(knn_matrix))

        # Print knn matrix
        plt.figure(figsize=(7, 7))
        plt.spy(knn_matrix, markersize=5)
        plt.ylabel("Users", fontsize=18)
        plt.title("KNN")
        plt.show()
        
        # Ask user if want to save the data for next runs
        if ask_yes_no("Do you want to cache this data?") == 'Y':
            path = 'amlsim_data_saved'

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

        os.makedirs(f'images/{dimension}/', exist_ok=True)

        if ask_yes_no("Do you want to visualize the results of SQUIC-Fit?") == 'Y':
            printMatrix = True

            # Print knn matrix
            plt.figure(figsize=(7, 7))
            plt.spy(knn_matrix, markersize=5)
            plt.ylabel("Users", fontsize=18)
            plt.title("KNN")
            plt.show()
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

    # Results
    dict_cluster = clustering_optimal_number(dimension, results_squic, plot=False)

    metrics = modularity_density(dict_cluster, results_squic, leiden=True)

    for rho, data in metrics.items():
        print(f"For rho {rho}:")
        for method, results in data.items():
            print(f"    {method}: PDensity = {results['p_density']}, Int Density = {results['int_density']}, Q = {results['modularity']}, nCluster = {results['nCluster']}, nIsolated  = {results['isolated']}")

    plot_PDens_Q(name, metrics, dimension, squic_method, save)
    