from workflow.internal import *


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    df, name, dimension = load_dataset()

    # Extract time series
    extract_timeseries(df)

    # Normalise time series
    Y_norm = normalization(df, name)

    # Run SQUIC_fit
    adjaceny_matrices, _, lambdas = squic_fit_computation(Y_norm, name, dimension)
    
    # Use the extracted W for clustering
    dict_cluster = clustering(adjaceny_matrices, lambdas)

    # Report internal metrics on the clustering
    int_metrics = internal_metrics(dict_cluster, adjaceny_matrices, lambdas)
    visualize_metrics(int_metrics, lambdas)

    # Visualise with cosmograph
    # visualize_graph(adjaceny_matrices, dict_cluster, lambdas, name, dimension)