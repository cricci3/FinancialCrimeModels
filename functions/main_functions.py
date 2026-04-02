import pandas as pd
import numpy as np
from scipy import sparse
import json
from functions.internal import normalization, load_dataset
from functions.plots import plot_timeseries, plot_knn
from functions.clustering_functions import study_CC
from functions.SQUIC_functions import squic_fit_matrix_computation, squic_fit_computation
from sklearn.neighbors import NearestNeighbors
import os


def show_df(Y):
    n_users, n_days = Y.shape

    # Create labels
    day_labels = [f"Day {i}" for i in range(n_days)]
    user_labels = list(range(n_users))

    # Create the DataFrame
    df = pd.DataFrame(Y, index=user_labels, columns=day_labels)

    print(df)


def load_presaved_data(dimension, name="PAYSIM"):
    path = f'{name}_data_saved'

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
            print("std dev not 1")
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
        
        # Load pre-saved account_properties
        with open(f'{path}/account_prop_{dimension}.json', 'r') as f:
            account_prop = json.load(f)

        account_prop = {int(k): v for k, v in account_prop.items()}
        
        return Y_norm, knn_matrix, account_prop, dimension

    except Exception as e:
        # If data are not present
        print(f"Error loading data: {e}")
        exit(1)


def normal_run(dimension, plot_fig, save_fig=False, cache_data=False, labels=False):
    Y, name, dimension, account_prop = load_dataset(valid_names={"PAYSIM"})

    # visualize timeseries
    if plot_fig:
        plot_timeseries(Y, name, dimension, labels=labels, save_fig=save_fig)

    if name == 'PAYSIM':
        print(Y)

        print(Y.shape)
        dim1, _ = Y.shape
        if dim1 == 365:
            Y = Y.T
    else:
        Y = Y.T.values  # Convert to (users, days)
        show_df(Y)


    Y_norm = normalization(Y, name)

    print(f"\n\n\nSize of Y Norm is {Y_norm.shape}\n\n\n")

    if name == 'PAYSIM':
        # visualize timeseries with normalized data
        plot_timeseries(Y_norm, name, dimension, save_fig=save_fig, type_df='norm', labels=labels, account_prop=account_prop)
        print(Y_norm)
    else:
        plot_timeseries(Y_norm, name, save_fig=save_fig, type_df='norm')
        show_df(Y_norm)

    std_dev = round(np.mean(np.std(Y_norm, axis=1)), 2)
    print("Max:", np.max(Y_norm))
    print("Min:", np.min(Y_norm))
    print("Mean std per row (should be 1):", std_dev)

    n_neigs = {
        "PAYSIM" : {
            '100' : 10,
            '1K' : 8,
            '10K' : 4,
            '100K' : 3, # can run even with 4
            '1M' : 2
        },
        "AMLSIM" : {
            '100' : 10,
            '1K' : 5,
            '10K' : 5,
            '100K' : 4,
            '1M' : 3
        }
    }

    nbrs = NearestNeighbors(n_neighbors=n_neigs[name][dimension], metric='euclidean', n_jobs=-1)
    nbrs.fit(Y_norm)
    knn_matrix = nbrs.kneighbors_graph(Y_norm, mode='connectivity')
    
    print("Shape KNN:", knn_matrix.shape)
    print("nnz KNN:", knn_matrix.nnz)

    # Ask user if want to save the data for next runs
    if cache_data:
        path = f'{name}_data_saved'

        # if path does not exists, create it
        os.makedirs(path, exist_ok=True)
        
        # Save Y_norm as CSV
        pd.DataFrame(Y_norm).to_csv(f'{path}/YNorm_{dimension}.csv', index=False)
        
        # Save account_prop as json
        with open(f'{path}/account_prop_{dimension}.json', 'w') as f:
            json.dump(account_prop, f)
        
        # Save knn_matrix as npz
        sparse.save_npz(f'{path}/knn_matrix_{dimension}.npz', knn_matrix)

    return Y_norm, knn_matrix, account_prop, dimension


def run_squic_fit_matrix(Y_norm, knn_matrix, results_squic, dimension, plot_fig=False, save_fig=False, study_cc=False):
    if plot_fig:
        plot_knn(knn_matrix, dimension, save=save_fig)

    if study_cc:
        # A plot showing the top 5 components of KNN matrix and SQUIC-Fit results will be show
        study_CC(knn_matrix)
    
    squic_method = 'squic-fit-matrix'
    results_squic[squic_method], times_dict = squic_fit_matrix_computation(Y_norm, dimension, knn_matrix, study_cc, plot_fig, save_fig)

    return results_squic, squic_method, times_dict


def run_squic_fit(Y_norm, results_squic, name, dimension, plot_fig=False, save_fig=False):
    squic_method = 'squic-fit'
    results_squic[squic_method], times_dict = squic_fit_computation(Y_norm, name, dimension, plot_fig, save_fig)
    return results_squic, squic_method, times_dict