from workflow.internal import extract_timeseries, normalization, adjaceny_matrix, squic_fit_computation, clustering, internal_metrics, visualize_metrics, visualize_graph
from workflow.external import load_dataset, squic_computation, linear_DA 


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    df, name, dimension, account_prop = load_dataset()

    # Extract time series
    extract_timeseries(df, name)

    # Normalise time series
    Y_norm = normalization(df, name)

    # Create transaction matrix
    adj_matrix = adjaceny_matrix(Y_norm, name, dimension)

    # Run SQUIC_fit to compute A (old W)
    A_matrices, _ = squic_fit_computation(Y_norm, name, dimension, adj_matrix, printMatrix=False)

    # Run SQUIC to compute Θ
    Theta_matrices, _ = squic_computation(Y_norm, name, dimension)

    # Use the extracted W for clustering
    dict_cluster = clustering(A_matrices)

    # extract Θ for LDA
    dict_scores, y = linear_DA(Theta_matrices, account_prop)

    # Report internal metrics on the clustering
    # int_metrics = internal_metrics(dict_cluster, W_matrices)
    # visualize_metrics(int_metrics)

    # Visualise with cosmograph
    # visualize_graph(W_matrices, dict_cluster, name, dimension)

    # Report external metrics on the classification of Θ
    #ext_metrics = external_metrics(dict_scores, y)
    #print(ext_metrics)

    # Visualize with cosmograph