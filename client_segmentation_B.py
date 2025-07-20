from functions.internal import load_dataset_1B, normalization
from functions.SQUIC_functions import squic_fit_matrix_computation, squic_fit_computation, check_symmetric_sparse
from functions.clustering_functions import clustering_optimal_number, modularity_density
import matplotlib.pyplot as plt
import json
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import os
from functions.plots import plot_timeseries, plot_knn, plot_PDens_Q
from functions.main_functions import load_presaved_data, normal_run, run_squic_fit, run_squic, show_df, ask_input, ask_yes_no


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
    name = 'AMLSIM'

    if user_input == 'Y':
        Y_norm, knn_matrix, account_prop, dimension = load_presaved_data(name)
    
    else: # Normal run
        Y_norm, knn_matrix, account_prop, dimension = normal_run(name)
    
    results_squic = {}

    # Run SQUIC_fit
    if ask_yes_no("SQUIC-Fit with bias or no?") == 'Y':
        results_squic, squic_method, save = run_squic_fit(Y_norm, knn_matrix, results_squic, name, dimension)

    else:
        results_squic, squic_method, save = run_squic(Y_norm, results_squic, name, dimension)
        
    # Results
    dict_cluster = clustering_optimal_number(dimension, results_squic, plot=False)

    metrics = modularity_density(dict_cluster, results_squic, leiden=True)

    for rho, data in metrics.items():
        print(f"For rho {rho}:")
        for method, results in data.items():
            print(f"    {method}: PDensity = {results['p_density']}, Int Density = {results['int_density']}, Q = {results['modularity']}, nCluster = {results['nCluster']}, nIsolated  = {results['isolated']}")

    plot_PDens_Q(name, metrics, dimension, squic_method, save)
    