from workflow.internal import extract_timeseries, normalization, adjaceny_matrix, squic_fit_computation, clustering, internal_metrics, visualize_metrics, visualize_graph_internal
from workflow.external import load_dataset, squic_computation, prepare_LDA , external_metrics, visualize_graph_external


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    Y, name, dimension, account_prop, trans_matrix = load_dataset()

    # Extract time series
    # extract_timeseries(Y, name)

    # Normalise time series
    Y_new = normalization(Y, name)

    # Run SQUIC to compute Θ
    Theta_matrices, _ = squic_computation(Y_new, name, dimension)

    # extract Θ for LDA
    ext_scores = prepare_LDA(Theta_matrices, account_prop)

    # Report external metrics on the classification of Θ
    ext_metrics = external_metrics(ext_scores)
    #print(ext_metrics)

    # Visualise with cosmograph Θ
    visualize_graph_external(Theta_matrices, account_prop, name, dimension)
