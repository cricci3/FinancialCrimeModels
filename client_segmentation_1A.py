from workflow.internal import load_dataset, normalization, visualize_metrics
from workflow.SQUIC_functions import squic_fit_matrix_computation
from workflow.clustering_functions import clustering_2_communities, modularity_fscore, plot_Q_f1


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    Y, name, dimension, account_prop, trans_matrix = load_dataset()

    # Extract time series
    # extract_timeseries(Y, name)

    # Normalise time series
    Y_norm = normalization(Y, name)

    results_squic = {}

    # Run SQUIC_fit
    results_squic['squic-fit-matrix'] = squic_fit_matrix_computation(Y_norm, name, dimension, trans_matrix)

    dict_cluster = clustering_2_communities(results_squic, name, account_prop)

    # Report internal metrics on the clustering
    metrics = modularity_fscore(dict_cluster, results_squic, account_prop)

    visualize_metrics(metrics)

    plot_Q_f1(metrics)
