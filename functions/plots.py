import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import os
import seaborn as sns


def plot_timeseries(df, name, dimension=None, type_df=None, print_fig=False, account_prop=None):
    # Plot the balance evolution for all users (columns) as separate lines
    # plt.figure(figsize=(15,10), dpi= 300)
    plt.figure(figsize=(9,7))

    if type_df == 'norm':
        df = df.T

        if name == 'AMLSIM':
            n_days, n_users = df.shape
            # Create labels
            user_labels = [f"User_{i}" for i in range(n_users)]
            day_labels = list(range(n_days))

            # Create the DataFrame
            df = pd.DataFrame(df, index=day_labels, columns=user_labels)

    if name == 'PAYSIM':
        # colors = {'Clients': 'mediumseagreen', 
        #       'Bank': 'crimson', 
        #       'Merchants': 'darkturquoise'}

        colors = {
            'C': '#88B06B',   # Clients
            'B': '#F0A631',   # Bank
            'M': '#D95F5F'    # Merchants
        }
        
        bank_dimension = ['100', '1K', '10K']

        # Plot each column (account balance) as a line
        for col in df.columns:
            if type_df == 'norm' and account_prop is not None:
                # map integer col → original user id and class
                user_info = account_prop[col]
                user_id = user_info["original_id"]
                user_class = user_info["class"][0]  # first char: 'C', 'M', or 'B'
            else:
                # no normalization: original names in df
                user_id = col
                user_class = col[0] if len(col) > 0 else 'C'
            
            # select color
            if user_class == 'C':
                color = colors['C']
                plt.plot(df.index, df[col], color=color, alpha=0.7)
            elif user_class == 'B':
                if dimension in bank_dimension:
                    color = colors['B']
                    plt.plot(df.index, df[col], color=color, alpha=0.7)
            elif user_class == 'M':
                color = colors['M']
                plt.plot(df.index, df[col], color=color, alpha=0.7)
            else:
                plt.plot(df.index, df[col], alpha=0.7)
            
        if dimension in bank_dimension:
            legend_elements = [
                Line2D([0], [0], color=colors['C'], lw=4, label='Clients'),
                Line2D([0], [0], color=colors['B'], lw=4, label='Bank'),
                Line2D([0], [0], color=colors['M'], lw=4, label='Merchants')
            ]
        else:
            legend_elements = [
                Line2D([0], [0], color=colors['C'], lw=4, label='Clients'),
                Line2D([0], [0], color=colors['M'], lw=4, label='Merchants')
            ]

        plt.legend(handles=legend_elements, fontsize=16)
    
    else:
        # Plot each column (account balance) as a line
        for user in df.columns:
            plt.plot(df.index, df[user], label=f"User {user}", alpha=0.7)

    plt.xlabel("Days", fontsize=18)
    plt.ylabel("Balance", fontsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.grid(axis='y', color='#cccccc', linewidth=0.5, alpha=0.3, linestyle='--')
    plt.grid(axis='x', color='#cccccc', linewidth=0.5, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if print_fig:
        if type_df is None:
            plt.savefig(f'ts_{name}_{dimension}', dpi=300)
        else:
            plt.savefig(f'ts_{name}_{dimension}_norm', dpi=300)
    plt.show()
    return


def plot_knn(knn_matrix, name, dimension, save=False):
    plt.figure(figsize=(7, 7))
    plt.spy(knn_matrix, markersize=5)
    plt.xlabel("Users", fontsize=18)
    plt.ylabel("Users", fontsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    if save:
        dir = f'images/{name}/{dimension}'
        os.makedirs(dir, exist_ok=True)
        plt.savefig(f'{dir}/knn_matrix')
    plt.show()
    return


def print_covariance_matrix(X, show=False, save=False, path=None, file_name=None):
    '''
    Function to print adjaceny matrix
    '''
    plt.figure(figsize=(7, 7))
    plt.spy(X, markersize=5)
    # plt.xlabel("Users", fontsize=18)
    plt.xlabel("Users", fontsize=18)
    plt.ylabel("Users", fontsize=18)

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)  

    if save:
        # if path does not exists, create it
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/{file_name}", dpi=180)
    
    if show:
        plt.show()
    return


def plot_ARI_f1(metrics_dict, squic_method, dimension, save=False):
    methods = [squic_method]
    colors = {
        squic_method: 'tab:orange'
    }

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    for method in methods:

        rhos = sorted(metrics_dict[method].keys())
        ARI_values = []
        F1_values = []
        valid_rhos = []

        for rho in rhos:
            metrics = metrics_dict[method][rho].get('Spectral', {})
            ari = metrics.get('ARI', None)
            f1 = metrics.get('f1', None)
            ARI_values.append(ari)
            F1_values.append(f1)
            valid_rhos.append(rho)

        color = colors[method]

        # Modularity Q — dotted line
        ax1.plot(valid_rhos, ARI_values, linestyle='dashed', marker='o', color='orange', label=f'ARI')

        # F1-score — solid line
        ax2.plot(valid_rhos, F1_values, linestyle='solid', marker='^', color='green', label=f'F1')

    # Axis labels
    # ax1.set_xlabel("Reg. parameter lambda", fontsize=18)
    # plt.rcParams['text.usetex'] = True  # Enable LaTeX
    ax1.set_xlabel(r'Reg. parameter $\lambda$', fontsize=18)
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
        os.makedirs(f'images/PAYSIM/{dimension}', exist_ok=True)
        plt.savefig(f"images/PAYSIM/{dimension}/ARI_F1_{squic_method}")
        print(f"Plot saved in images/PAYSIM/{dimension}/ARI_F1_{squic_method}")
    plt.show()
    return


def plot_PDens_Q(name, metrics_dict, dimension, squic_method, save=False, metric='q'):

    colors = {
        'Louvain':'green',
        'Leiden':'orange',
        'Spectral':'cornflowerblue',
        'DBSCAN':'mediumorchid'
    }

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    # Get the list of methods from the first rho entry
    first_rho = next(iter(metrics_dict))
    clustering_methods = metrics_dict[first_rho].keys()

    for method in clustering_methods:
        PDensity_values = []
        second_values = []
        valid_rhos = []

        for rho in sorted(metrics_dict.keys()):
            method_metrics = metrics_dict[rho].get(method, {})
            pdens = method_metrics.get('p_density', None)
            if metric == 'q':
                second_metric = method_metrics.get('modularity', None)
            elif metric == 'int_density':
                second_metric = method_metrics.get('int_density', None)


            if pdens is not None and second_metric is not None:
                PDensity_values.append(pdens)
                second_values.append(second_metric)
                valid_rhos.append(rho)

        if valid_rhos:
            color = colors.get(method, 'black')
            if metric == 'q':
                ax1.plot(valid_rhos, second_values, linestyle='dashed', marker='o', color=color, label=f'Q {method}')
            elif metric == 'int_density':
                ax1.plot(valid_rhos, second_values, linestyle='dashed', marker='o', color=color, label=f'IntDensity {method}')
            ax2.plot(valid_rhos, PDensity_values, linestyle='solid', marker='^', color=color, label=f'Pdensity {method}')

    # ax1.set_xlabel("Reg. Parameter lambda", fontsize=18)
    # plt.rcParams['text.usetex'] = True  # Enable LaTeX
    ax1.set_xlabel(r'Reg. parameter $\lambda$', fontsize=18)
    if metric == 'q':
        ax1.set_ylabel("Modularity Q", color='black', fontsize=18)
    elif metric == 'int_density':
        ax1.set_ylabel("Internal Density", color='black', fontsize=18)
    ax2.set_ylabel("Partition Density", color='black', fontsize=18)

    ax1.set_xticks(valid_rhos)
    ax1.set_xticklabels([str(r) for r in valid_rhos], fontsize=18)

    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=12)

    fig.tight_layout()
    plt.grid(True)

    if save:
        os.makedirs(f'images/AMLSIM/{name}/{dimension}', exist_ok=True)
        plt.savefig(f"images/AMLSIM/{name}/{dimension}/{squic_method}-PDens_Q.png")
        print(f"Plot saved in images/AMLSIM/{dimension} image {squic_method}-PDens_Q.png")

    plt.show()
    return


def plot_CC(group_coords, colors):
    plt.figure(figsize=(7, 7))
    for group_id, (rows, cols) in group_coords.items():
        plt.scatter(cols, rows, s=0.5, color=colors[group_id], label=f'Component {group_id}' if group_id >= 0 else 'Other', alpha=0.6)

    plt.xlabel("Users", fontsize=18)
    plt.ylabel("Users", fontsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.legend(markerscale=6, fontsize=12, loc='upper right')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    return


def plot_results(metrics, ylabel, results, methods=['louvain', 'spectral', 'dbscan']):
    # Create the plot
    plt.figure(figsize=(6, 6))

    colors = ['steelblue', 'gold', 'indianred']
    dot = ['o-', 's-', '^-']

    if methods == ['louvain', 'spectral']:
        for i, method in enumerate(methods): 
            plt.plot(results['rho_values'], results[f'{method}_{metrics}'], dot[i], label=f'{method}', color=colors[i])
        
        plt.yscale('log')
    
    else:
        plt.plot(results['rho_values'], results[f'louvain_{metrics}'], 'o-', label='Louvain', color=colors[0])
        plt.plot(results['rho_values'], results[f'spectral_{metrics}'], 's-', label='Spectral', color=colors[1])
        plt.plot(results['rho_values'], results[f'dbscan_{metrics}'], '^-', label='DBSCAN', color=colors[2])

    # plt.xlabel('lambda', fontsize=22)
    # plt.rcParams['text.usetex'] = True  # Enable LaTeX
    plt.xlabel(r'Reg. parameter $\lambda$', fontsize=18)
    plt.ylabel(ylabel, fontsize=22)
    plt.xscale('log')

    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.legend()
    plt.show()


def print_transaction_matrix(matrix):
    mask = (matrix == 0) # cover all zeros

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap='YlGnBu', fmt=".2f", cbar=True, mask=mask,
                linewidths=0.5, linecolor='white')

    plt.xlabel("Receiver Account", fontsize=22)
    plt.ylabel("Sender Account", fontsize=22)
    plt.show()

    return