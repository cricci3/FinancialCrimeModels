from workflow.internal import load_dataset, normalization, visualize_metrics
from workflow.SQUIC_functions import squic_computation, squic_matrix_computation, squic_fit_computation, squic_fit_matrix_computation
from workflow.clustering_functions import clustering_2_communities, modularity_fscore
import matplotlib.pyplot as plt


def plot_Q_f1(metrics_dict):
    methods = ['squic', 'squic-matrix', 'squic-fit', 'squic-fit-matrix']
    colors = {
        'squic': 'tab:blue',
        'squic-matrix': 'tab:orange',
        'squic-fit': 'tab:green',
        'squic-fit-matrix': 'tab:purple'
    }

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for method in methods:

        rhos = sorted(metrics_dict[method].keys())
        Q_values = []
        F1_values = []
        valid_rhos = []

        for rho in rhos:
            metrics = metrics_dict[method][rho].get('spectral', {})
            Q = metrics.get('modularity', None)
            f1 = metrics.get('f1', None)
            Q_values.append(Q)
            F1_values.append(f1)
            valid_rhos.append(rho)

        color = colors[method]

        # Modularity Q — dotted line
        ax1.plot(valid_rhos, Q_values, linestyle='dotted', color=color, label=f'{method} Q')

        # F1-score — solid line
        ax2.plot(valid_rhos, F1_values, linestyle='solid', color=color, label=f'{method} F1')

    # Axis labels
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("Modularity Q", color='black')
    ax2.set_ylabel("F1 Score", color='black')

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    fig.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    Y, name, dimension, account_prop, trans_matrix = load_dataset()

    # Extract time series
    # extract_timeseries(Y, name)

    # Normalise time series
    Y_norm = normalization(Y, name)

    results_squic = {}

    # Run SQUIC_fit
    results_squic['squic'] = squic_computation(Y_norm, name, dimension)
    results_squic['squic-matrix'] = squic_matrix_computation(Y_norm, name, dimension, trans_matrix)
    results_squic['squic-fit'] = squic_fit_computation(Y_norm, name, dimension)
    results_squic['squic-fit-matrix'] = squic_fit_matrix_computation(Y_norm, name, dimension, trans_matrix)

    dict_cluster = clustering_2_communities(results_squic, name, account_prop)

    # Report internal metrics on the clustering
    metrics = modularity_fscore(dict_cluster, results_squic, account_prop)

    visualize_metrics(metrics)

    plot_Q_f1(metrics)




