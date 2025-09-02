import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import os
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_timeseries(df, name, dimension=None, labels=False, type_df=None, save_fig=None, account_prop=None, title=None):
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

    if name == 'PAYSIM' and labels == True:
        # colors = {'Clients': 'mediumseagreen', 
        #       'Bank': 'crimson', 
        #       'Merchants': 'darkturquoise'}

        colors = {
            'C': '#88B06B',   # Clients
            'B': '#F0A631',   # Bank
            'M': '#D95F5F'    # Merchants
        }
        
        bank_dimension = ['100', '1K']

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
            if name == "PAYSIM" and str(user)[0] == 'B':
                pass
            else:
                plt.plot(df.index, df[user], label=f"User {user}", alpha=0.7)

    plt.xlabel("Days", fontsize=18)
    plt.ylabel("Balance", fontsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.grid(axis='y', color='#cccccc', linewidth=0.5, alpha=0.3, linestyle='--')
    plt.grid(axis='x', color='#cccccc', linewidth=0.5, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_fig is not None and save_fig != False:
        os.makedirs(f'{save_fig}/{dimension}', exist_ok=True)
        if type_df is None:
            plt.savefig(f'{save_fig}/{dimension}/{title}', dpi=300)
        else:
            plt.savefig(f'{save_fig}/{dimension}/{title}', dpi=300)
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


def plot_theta(theta, name, dim, LAMBDA, save=False):
    plt.figure(figsize=(7, 7))
    plt.spy(theta, markersize=5)
    plt.xlabel("Users", fontsize=18)
    plt.ylabel("Users", fontsize=18)

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18) 
    if save:
        path = f'decomposition/glasso_images/{name.lower()}/{dim}'
        os.makedirs(path, exist_ok=True)
        lambda_str = str(LAMBDA).replace(".", "")
        plt.savefig(f"{path}/squic_fit_{lambda_str}", dpi=300)
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
        plt.savefig(f"{path}/{file_name}", dpi=300)
    
    if show:
        plt.show()
    return


# def plot_ARI_f1(metrics_dict, squic_method, dimension, times_dict=None, save=False):
#     methods = [squic_method]
#     colors = {squic_method: 'tab:orange'}

#     fig, ax1 = plt.subplots(figsize=(8, 8))
#     ax2 = ax1.twinx()

#     for method in methods:
#         rhos = sorted(metrics_dict[method].keys())
#         ARI_values, F1_values = [], []

#         for rho in rhos:
#             metrics = metrics_dict[method][rho].get('Spectral', {})
#             ARI_values.append(metrics.get('ARI', None))
#             F1_values.append(metrics.get('f1', None))

#         color = colors[method]
#         x_positions = range(len(rhos))  # equally spaced positions

#         # ARI — dashed line
#         ax1.plot(x_positions, ARI_values, linestyle='dashed', marker='o',
#                  color='orange', label='ARI')

#         # F1-score — solid line
#         ax2.plot(x_positions, F1_values, linestyle='solid', marker='^',
#                  color='green', label='F1')

#         # Replace numeric ticks with rho labels
#         ax1.set_xticks(x_positions)
#         ax1.set_xticklabels([str(r) for r in rhos], fontsize=13)

#     ax1.tick_params(axis='y', labelsize=13)
#     ax2.tick_params(axis='y', labelsize=13)

#     # Labels
#     ax1.set_xlabel(r'Reg. parameter $\lambda$', fontsize=18)
#     ax1.set_ylabel("ARI", color='black', fontsize=18)
#     ax2.set_ylabel("F1 Score", color='black', fontsize=18)

#     # Legends
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines1 + lines2, labels1 + labels2,
#                loc='upper center', bbox_to_anchor=(0.5, -0.2),
#                ncol=2, fontsize=14, frameon=True)

#     fig.tight_layout()
#     plt.grid(True, axis='y', alpha=0.2)
#     if save:
#         os.makedirs(f'images/PAYSIM/{dimension}', exist_ok=True)
#         plt.savefig(f"images/PAYSIM/{dimension}/ARI_F1_{squic_method}")
#         print(f"Plot saved in images/PAYSIM/{dimension}/ARI_F1_{squic_method}")
#     plt.show()
#     return

def plot_ARI_f1(metrics_dict, squic_method, dimension, times_dict=None, save=False):
    methods = [squic_method]
    colors = {squic_method: 'tab:orange'}

    # Create figure with 2 rows if times_dict is provided
    if times_dict is not None:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True
        )
        ax1 = ax_top
        ax2 = ax1.twinx()
    else:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
        ax_bottom = None

    for method in methods:
        rhos = sorted(metrics_dict[method].keys())
        ARI_values, F1_values = [], []

        for rho in rhos:
            metrics = metrics_dict[method][rho].get('Spectral', {})
            ARI_values.append(metrics.get('ARI', None))
            F1_values.append(metrics.get('f1', None))

        x_positions = range(len(rhos))  # equally spaced positions

        # ARI — dashed line
        ax1.plot(x_positions, ARI_values, linestyle='dotted', marker='o',
                 color='orange', label='ARI', linewidth=1.5)

        # F1-score — solid line
        ax2.plot(x_positions, F1_values, linestyle='solid', marker='^',
                 color='green', label='F1', linewidth=1.5)

        # Replace numeric ticks with rho labels
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels([str(r) for r in rhos], fontsize=13)

        # If times_dict provided, add bar plot below
        if times_dict is not None and ax_bottom is not None:
            times = [times_dict.get(rho, 0) for rho in rhos]
            ax_bottom.bar(x_positions, times, width=0.6, color='steelblue', alpha=0.7)
            ax_bottom.set_ylabel('SQUIC-Fit Time (s)', fontsize=16)
            ax_bottom.tick_params(axis='y', labelsize=13)
            ax_bottom.tick_params(axis='x', labelsize=13)
            ax_bottom.grid(True, axis='y', alpha=0.2)
            ax_bottom.set_xlabel(r'Reg. parameter $\lambda$', fontsize=18)  # x-label only on bottom

    # Labels
    ax1.set_ylabel("ARI", color='black', fontsize=18)
    ax2.set_ylabel("F1 Score", color='black', fontsize=18)

    if times_dict is None:  # if no time subplot, put xlabel on main plot
        ax1.set_xlabel(r'Reg. parameter $\lambda$', fontsize=18)

    ax1.tick_params(axis='y', labelsize=13)
    ax2.tick_params(axis='y', labelsize=13)

    # Legends (above ARI/F1 plot)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, fontsize=14, frameon=True)
    
    plt.grid(True, axis='y', alpha=0.2)

    fig.tight_layout()
    if save:
        path = f'images/PAYSIM/{dimension}'
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/ARI_F1_{squic_method}", dpi=300)
        print(f"Plot saved in {path} as ARI_F1_{squic_method}")
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


def plot_PDens_Q_Times(int_metrics, dim, name, save=False):
    colors = {
        'louvain': 'green',
        'leiden': 'orange',
        'spectral': 'cornflowerblue',
        'dbscan': 'mediumorchid'
    }

    # figure with a short time panel below
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])          # Q (left y)
    ax2 = ax1.twinx()                         # Pdensity (right y)
    ax_time = fig.add_subplot(gs[1, 0], sharex=ax1)

    # Collect the sorted unique rho values (λ) for x (equally spaced categories)
    all_rhos = sorted(int_metrics.keys())
    x_pos_map = {rho: i for i, rho in enumerate(all_rhos)}  # rho -> index 0..N-1

    # Methods present in the first rho entry — keep only dict-valued keys (exclude 'time')
    first_rho = all_rhos[0]
    clustering_methods = [k for k, v in int_metrics[first_rho].items() if isinstance(v, dict)]

    # --- Time panel data (seconds) ---
    time_values, time_x = [], []
    for rho in all_rhos:
        end_time = int_metrics.get(rho, {}).get('time', np.nan)  # float seconds
        # if some entry is missing or NaN, keep NaN (line plot will skip; bar will show 0 if we choose to)
        time_values.append(end_time)
        time_x.append(x_pos_map[rho])

    # --- Top panel: Q and Pdensity for each method ---
    for method in clustering_methods:
        x_idx, q_vals, pden_vals = [], [], []
        for rho in all_rhos:
            mm = int_metrics.get(rho, {}).get(method, None)
            if not isinstance(mm, dict):
                continue
            pdens = mm.get('p_density', None)
            modularity = mm.get('modularity', None)
            if (pdens is not None) and (modularity is not None):
                x_idx.append(x_pos_map[rho])
                q_vals.append(modularity)
                pden_vals.append(pdens)

        if x_idx:
            c = colors.get(method, 'black')
            ax1.plot(x_idx, q_vals, linestyle='dotted', marker='o', color=c, label=f'Q {method}', linewidth=1.5)
            ax2.plot(x_idx, pden_vals, linestyle='-', marker='^', color=c, label=f'Pdensity {method}', linewidth=1.5)

    # ===== top axes labels & ticks =====
    ax1.set_ylabel('Modularity Q', color='black', fontsize=16)
    ax2.set_ylabel('Partition Density', color='black', fontsize=16)
    ax1.set_xticks(range(len(all_rhos)))
    ax1.set_xticklabels([str(r) for r in all_rhos], fontsize=13)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax1.grid(True, axis='y', alpha=0.3)

    # unified legend (top plot)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
            loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=4, fontsize=11, frameon=True)

    # ===== bottom time panel (seconds) =====
    time_values = np.array(time_values, dtype=float)
    # bars (replace NaN with 0 for bars; or switch to a line plot to skip NaNs)
    ax_time.bar(time_x, np.nan_to_num(time_values, nan=0.0),
                width=0.6, alpha=0.6, linewidth=0.5)

    ax_time.set_ylabel('SQUIC-Fit Time (s)', fontsize=14)
    ax_time.set_xlabel(r'Reg. parameter $\lambda$', fontsize=16)
    ax_time.tick_params(axis='x', labelsize=13)
    ax_time.tick_params(axis='y', labelsize=13)
    ax_time.grid(True, axis='y', alpha=0.2)

    # hide duplicated x tick labels on top axis
    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.tight_layout()
    if save:
        path = f'decomposition/glasso_images/{name.lower()}/{dim}'
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/plot', dpi=300)
    plt.show()


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