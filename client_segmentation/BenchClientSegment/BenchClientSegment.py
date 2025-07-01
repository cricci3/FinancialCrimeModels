from functions.internal import *
from functions.clustering_functions import *


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    Y, name, dimension, account_prop, trans_matrix = load_dataset()

    # Extract time series
    extract_timeseries(Y, name)

    # Normalise time series
    Y_new = normalization(Y, name)

    # Print transaction matrix matrix
    # print_transaction_matrix(trans_matrix)

    # Run SQUIC_fit
    W_matrices, _ = squic_fit_computation(Y_new, name, dimension, trans_matrix, printMatrix=True)
    
    # Use the extracted W for clustering
    dict_cluster = clustering_same_n(W_matrices)

    # Report internal metrics on the clustering
    int_metrics = internal_metrics(dict_cluster, W_matrices)
    visualize_metrics(int_metrics)