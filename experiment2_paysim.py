from functions.clustering_functions import clustering_2_communities, ARI_fscore
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
from functions.plots import plot_ARI_f1
from functions.main_functions import load_presaved_data, normal_run, run_squic_fit, run_squic_fit_matrix, ask_input, ask_yes_no
import yaml


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
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    cache = config['load_cache']
    dimension = config['dataset_dimension']
    save_fig = config['save_fig']
    plot_fig = config['plot_fig']
    cache_data = config['cache_data']
    labels = config['labels']

    bias_squic = config['squic_fit_options']['bias']
    plot_squic = config['squic_fit_options']['visualize_results']
    save_squic = config['squic_fit_options']['save_results']
    study_cc = config['squic_fit_options']['study_cc']

    if cache:
        Y_norm, knn_matrix, account_prop, dimension = load_presaved_data(dimension)
        
    else: # Normal run
        Y_norm, knn_matrix, account_prop, dimension = normal_run(
            name="PAYSIM",
            dimension=dimension,
            plot_fig=plot_fig,
            save_fig=save_fig,
            cache_data=cache_data,
            labels=True)
        
    results_squic = {}

    # Run SQUIC_fit
    if bias_squic:
        results_squic, squic_method, times_dict = run_squic_fit_matrix(Y_norm, knn_matrix, results_squic, dimension, plot_fig=plot_squic, save_fig=save_squic, study_cc=study_cc)
    else:
        results_squic, squic_method, times_dict = run_squic_fit(Y_norm, results_squic, name="PAYSIM", dimension=dimension)
        
    # Results
    dict_cluster = clustering_2_communities(results_squic, squic_method, dimension)

    metrics = ARI_fscore(dict_cluster, results_squic, account_prop)

    for _, data in metrics.items():
        for rho, results in data.items():
            print(f"for rho = {rho} : {results}")

    plot_ARI_f1(metrics, squic_method, dimension, times_dict, save_fig)
    