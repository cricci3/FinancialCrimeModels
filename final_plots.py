import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
import os


def plot_ARI_f1_conclusive(paysim_values, save=False):

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    ARI_values = []
    F1_values = []
    dimensions = []

    for key, _ in paysim_values.items():
        metrics = paysim_values.get(key)
        ari = metrics.get('ARI', None)
        f1 = metrics.get('f1', None)
        ARI_values.append(ari)
        F1_values.append(f1)
        dimensions.append(key)

    # ARI — dotted line
    ax1.plot(dimensions, ARI_values, linestyle='dashed', marker='o', color='orange', label=f'ARI')

    # F1-score — solid line
    ax2.plot(dimensions, F1_values, linestyle='solid', marker='^', color='green', label=f'F1')

    # Axis labels
    ax1.set_xlabel("Number of Users", color="black", fontsize=18)
    ax1.set_ylabel("ARI", color='black', fontsize=18)
    ax2.set_ylabel("F1 Score", color='black', fontsize=18)

    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)

    fig.tight_layout()
    plt.grid(True)
    if save:
        # if path does not exists, create it
        os.makedirs(f'images/PAYSIM', exist_ok=True)
        plt.savefig(f"images/PAYSIM/ARI_F1_conclusive")
        print(f"Plot saved in images/PAYSIM/ARI_F1_conclusive")
    plt.show()
    return


def plot_PDens_Q_conclusive(amlsim_values, metric='Q', save=False):
    colors = {
        'Louvain': 'green',
        'Leiden': 'orange',
        'Spectral': 'cornflowerblue',
        'DBSCAN': 'mediumorchid'
    }

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Get the list of methods from the first dataset size
    first_key = next(iter(amlsim_values))
    clustering_methods = amlsim_values[first_key].keys()

    # Ensure keys are sorted in order of size
    def sort_key(x):
        return int(x.replace('K', '000').replace('100', '100'))

    sorted_dims = sorted(amlsim_values.keys(), key=sort_key)

    for method in clustering_methods:
        PDensity_values = []
        second_metric_values = []
        valid_dims = []

        for dim in sorted_dims:
            method_metrics = amlsim_values[dim].get(method, {})
            pdens = method_metrics.get('PDensity', None)
            second_metric = method_metrics.get(metric, None)

            if pdens is not None and metric is not None:
                PDensity_values.append(pdens)
                second_metric_values.append(second_metric)
                valid_dims.append(dim)

        if valid_dims:
            color = colors.get(method, 'black')
            ax1.plot(valid_dims, second_metric_values, linestyle='dashed', marker='o', color=color, label=f'{method} {metric}')
            ax2.plot(valid_dims, PDensity_values, linestyle='solid', marker='^', color=color, label=f'{method} PDensity')

    ax1.set_xlabel("Dataset Size", fontsize=14)
    if metric == 'Q':
        ax1.set_ylabel("Modularity Q", color='black', fontsize=18)
    else:
        ax1.set_ylabel(f"{metric}", color='black', fontsize=18)
    ax2.set_ylabel("Partition Density", color='black', fontsize=18)

    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=12)

    fig.tight_layout()
    plt.grid(True)

    if save:
        os.makedirs(f'images/AMLSIM', exist_ok=True)
        plt.savefig(f"images/AMLSIM/PDens_{metric}_conclusive.png")
        print(f"Plot saved to images/AMLSIM/PDens_{metric}_conclusive.png")

    plt.show()



if __name__ == '__main__':
    paysim_values = {
        '100' : {'ARI': 1.0, 'f1': 1.0},
        '1K' : {'ARI': 0.97, 'f1': 0.99},
        '10K' : {'ARI': 0.92, 'f1': 0.98},
        '100K' : {'ARI': 0.74, 'f1': 0.93}
    }

    amlsim_values = {
        '100': {
            'Louvain': {'PDensity': 0.48, 'Int Density': 0.55, 'Q': 0.52, 'nCluster': 4},
            'Leiden': {'PDensity': 0.55, 'Int Density': 0.59, 'Q': 0.52, 'nCluster': 4},
            'DBSCAN': {'PDensity': 0.36, 'Int Density': 0.13, 'Q': 0.46, 'nCluster': 6},
            'Spectral': {'PDensity': 0.35, 'Int Density': 0.38, 'Q': 0.47, 'nCluster': 2}
        },
        '1K': {
            'Louvain': {'PDensity': 0.06, 'Int Density': 0.37, 'Q': 0.84, 'nCluster': 65},
            'Leiden': {'PDensity': 0.09, 'Int Density': 0.36, 'Q': 0.85, 'nCluster': 73},
            'DBSCAN': {'PDensity': 0.01, 'Int Density': 0.02, 'Q': 0.45, 'nCluster': 74},
            'Spectral': {'PDensity': 0.01, 'Int Density': 0.57, 'Q': 0.01, 'nCluster': 3}
        },
        '10K': {
            'Louvain': {'PDensity': 0.02, 'Int Density': 0.24, 'Q': 0.88, 'nCluster': 172},
            'Leiden': {'PDensity': 0.03, 'Int Density': 0.2, 'Q': 0.9, 'nCluster': 198},
            'DBSCAN': {'PDensity': 0.01, 'Int Density': 0.07, 'Q': 0.04, 'nCluster': 634},
            'Spectral': {'PDensity': 0.0, 'Int Density': 0.37, 'Q': 0.01, 'nCluster': 7}
        },
        '100K': {
            'Louvain': {'PDensity': 0.02, 'Int Density': 0.4, 'Q': 0.92, 'nCluster': 4693},
            'Leiden': {'PDensity': 0.02, 'Int Density': 0.4, 'Q': 0.94, 'nCluster': 4749},
            'DBSCAN': {'PDensity': 0.01, 'Int Density': 0.05, 'Q': 0.04, 'nCluster': 9256},
            'Spectral': {'PDensity': 0.0, 'Int Density': 0.0, 'Q': 0.04, 'nCluster': 2}
        }
    }

    plot_ARI_f1_conclusive(paysim_values, save=True)
    plot_PDens_Q_conclusive(amlsim_values, metric='Int Density', save=True)