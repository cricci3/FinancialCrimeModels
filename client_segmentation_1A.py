from functions.internal import load_dataset_1A, normalization
from functions.SQUIC_functions import squic_fit_matrix_computation, squic_fit_computation, check_symmetric_sparse
from functions.clustering_functions import clustering_2_communities, ARI_fscore, study_CC
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
import json
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import os
from functions.plots import plot_timeseries, plot_knn, plot_ARI_f1

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
    - The data must have been saved from a previous successful run using the same dimension.
    - On the first run for a given dimension, you must generate and save this data by answering 
      "Y" when prompted to cache it.
    '''
    user_input = ask_yes_no("\nDo you want to load data?")

    if user_input == 'Y':
        dimension = ask_input("Which dimension (100/1K/10K/100K/1M)?").upper()
        path = 'paysim_data_saved'

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

            if not(std_dev > 0.9 and std_dev <= 1):
                print("Mean std per row (should be 1):", std_dev)
                print("Max:", np.max(Y_norm))
                print("Min:", np.min(Y_norm))

                # Step 1: Convert DataFrame to NumPy array for performance
                Y = Y_norm.to_numpy()  # shape: (num_users, num_days)

                # Step 2: Compute row-wise standard deviation
                stds = np.std(Y, axis=1)
                safe_stds = np.clip(stds, 1e-8, None)  # Avoid division by near-zero

                # Step 3: Normalize each row by its std using broadcasting
                Y = Y / safe_stds[:, np.newaxis]

                Y_norm = pd.DataFrame(Y, index=Y_norm.index, columns=Y_norm.columns)

                print(f"Head of Y_norm df:")
                print(Y_norm.head())

                print("Mean std after normalization (should be ~1):", np.mean(np.std(Y_norm, axis=1)))
                print("Max:", np.max(Y_norm))
                print("Min:", np.min(Y_norm))
            else:
                print("Mean std per row (should be 1):", std_dev)

            # Load pre-saved knn_matrix
            knn_matrix = sparse.load_npz(f'{path}/knn_matrix_{dimension}.npz')

            print("Shape KNN:", knn_matrix.shape)
            print("nnz KNN:", knn_matrix.nnz)

            # Average number of non-zeros per row
            nnz_per_row = knn_matrix.nnz / knn_matrix.shape[0]
            print("nnz per row:", round(nnz_per_row, 2))

            print("Is symmetric:", check_symmetric_sparse(knn_matrix))
            
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
        Y, name, dimension, account_prop = load_dataset_1A()

        # visualize timeseries
        plot_timeseries(Y, name, dimension)

        print(Y)

        print(Y.shape)
        dim1, dim2 = Y.shape
        if dim1 == 365:
            Y = Y.T

        Y_norm = normalization(Y, name)

        # visualize timeseries with normalized data
        plot_timeseries(Y_norm, name, dimension, type_df='norm')

        print(Y_norm)

        print("NaNs in df?", np.isnan(Y_norm.values).any())
        print("Infs in df?", np.isinf(Y_norm.values).any())
        print("Max:", np.max(Y_norm))
        print("Min:", np.min(Y_norm))
        print("Mean std per row (should be 1):", round(np.mean(np.std(Y_norm, axis=1)), 2))

        n_neigs = {
            '100' : 10,
            '1K' : 8,
            '10K' : 4,
            '100K' : 3, # can run even with 4
            '1M' : 2 # then remove self loops
        }

        nbrs = NearestNeighbors(n_neighbors=n_neigs[dimension], metric='euclidean', n_jobs=-1)
        nbrs.fit(Y_norm)
        knn_matrix = nbrs.kneighbors_graph(Y_norm, mode='connectivity')

        # if dimension == '1M':
        #     # Remove self-loops
        #     knn_matrix.setdiag(0)
        #     knn_matrix.eliminate_zeros()
        
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

        os.makedirs(f'images/{dimension}/', exist_ok=True)

        plot_knn(knn_matrix, dimension)

        if ask_yes_no("Do you want to study the CC of KNN matrix and SQUIC-Fit results?") == 'Y':
            # A plot showing the top 5 components of KNN matrix and SQUIC-Fit results will be show
            study = True
            study_CC(knn_matrix)

        if ask_yes_no("Do you want to visualize the results of SQUIC-Fit?") == 'Y':
            printMatrix = True
        else:
            printMatrix = False

        if ask_yes_no("Do you want to save the results of SQUIC-Fit?") == 'Y':
            save = True
        else:
            save = False
        
        squic_method = 'squic-fit-matrix'
        results_squic[squic_method] = squic_fit_matrix_computation(Y_norm, name, dimension, knn_matrix, study, printMatrix, save)
    
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
    dict_cluster = clustering_2_communities(results_squic, squic_method)

    metrics = ARI_fscore(dict_cluster, results_squic, account_prop)

    for _, data in metrics.items():
        for rho, results in data.items():
            print(f"for rho = {rho} : {results}")

    plot_ARI_f1(metrics, squic_method, dimension, save)
    