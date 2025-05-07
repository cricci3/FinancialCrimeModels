from workflow.internal import *


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    df, name, dimension, account_prop = load_dataset()

    # Extract time series
    extract_timeseries(df, name)

    # Normalise time series
    Y_norm = normalization(df, name)

    # Create transaction matrix
    adj_matrix = adjaceny_matrix(Y_norm, name, dimension)

    # Run SQUIC_fit
    W_matrices, _ = squic_fit_computation(Y_norm, name, dimension, adj_matrix, printMatrix=False)
    
    # Use the extracted W for clustering
    dict_cluster = clustering(W_matrices)

    # Report internal metrics on the clustering
    int_metrics = internal_metrics(dict_cluster, W_matrices)
    visualize_metrics(int_metrics)

    # Visualise with cosmograph -> not working in .py file
    # visualize_graph(W_matrices, dict_cluster, account_prop, name, dimension)