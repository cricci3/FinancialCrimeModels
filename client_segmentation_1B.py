from workflow.internal import *
from workflow.clustering_functions import *


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    Y, name, dimension, account_prop, trans_matrix = load_dataset()

    # Extract time series
    # extract_timeseries(Y, name)

    # Normalise time series
    Y_new = normalization(Y, name)

    # Run SQUIC_fit
    W_matrices = squic_fit_matrix_computation(Y_new, name, dimension, trans_matrix, printMatrix=False)

    # Use the extracted W for clustering
    dict_cluster = clustering_optimal_number(W_matrices)

    # Report internal metrics on the clustering
    int_metrics = internal_metrics(dict_cluster, W_matrices, leiden=True)
    visualize_metrics(int_metrics)