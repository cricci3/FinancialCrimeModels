from workflow.internal import load_dataset, normalization
from workflow.SQUIC_functions import squic_fit_computation
from workflow.clustering_functions import clustering_2_communities, modularity_fscore, plot_Q_f1


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    Y, name, dimension, account_prop, trans_matrix = load_dataset()

    # Extract time series
    # extract_timeseries(Y, name)

    # Normalise time series
    Y_norm = normalization(Y, name)

    results_squic = {}

    # Run SQUIC_fit without bias
    results_squic['squic-fit-matrix'] = squic_fit_computation(Y_norm, name, dimension, printMatrix=False)

    dict_cluster = clustering_2_communities(results_squic, name, account_prop)

    metrics = modularity_fscore(dict_cluster, results_squic, account_prop)

    for _, data in metrics.items():
        for rho, results in data.items():
            print(f"for rho = {rho} : {results}")

    plot_Q_f1(metrics)
