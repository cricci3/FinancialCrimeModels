from functions.clustering_functions import clustering_2_communities, ARI_fscore
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
from functions.plots import plot_ARI_f1
from functions.main_functions import load_presaved_data, normal_run, run_squic_fit, run_squic, ask_input, ask_yes_no


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
    name = 'PAYSIM'

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
    dict_cluster = clustering_2_communities(results_squic, squic_method, name, dimension)

    metrics = ARI_fscore(dict_cluster, results_squic, account_prop)

    for _, data in metrics.items():
        for rho, results in data.items():
            print(f"for rho = {rho} : {results}")

    plot_ARI_f1(metrics, squic_method, dimension, save)
    