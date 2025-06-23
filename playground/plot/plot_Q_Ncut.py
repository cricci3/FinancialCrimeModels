import matplotlib.pyplot as plt

results = {
    '100': {
        'louvain': {'NCUT': 1.8, 'Q': 0.45, 'nCluster': 7},
        'leiden': {'NCUT': 2.0, 'Q': 0.41, 'nCluster': 7},
        'dbscan': {'NCUT': 43.52, 'Q': 0.2, 'nCluster': 47},
        'spectral': {'NCUT': 0.28, 'Q': 0.36, 'nCluster': 2}
    },
    '1K': {
        'louvain': {'NCUT': 2.03, 'Q': 0.68, 'nCluster': 157},
        'leiden': {'NCUT': 2.98, 'Q': 0.66, 'nCluster': 161},
        'dbscan': {'NCUT': 230.68, 'Q': 0.29, 'nCluster': 378},
        'spectral': {'NCUT': 0.35, 'Q': 0.52, 'nCluster': 3}
    },
    '10K': {
        'louvain': {'NCUT': 2.64, 'Q': 0.82, 'nCluster': 388},
        'leiden': {'NCUT': 5.02, 'Q': 0.84, 'nCluster': 407},
        'dbscan': {'NCUT': 284.24, 'Q': 0.22, 'nCluster': 612},
        'spectral': {'NCUT': 0.0, 'Q': 0.2, 'nCluster': 6}
    }
}

methods = ['louvain', 'leiden', 'dbscan', 'spectral']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, method in enumerate(methods):
    x = []      # Number of clusters
    y_q = []    # Q values
    y_ncut = [] # NCUT values
    
    for size in ['100', '1K', '10K']:
        entry = results[size][method]
        x.append(entry['nCluster'])
        y_q.append(entry['Q'])
        y_ncut.append(entry['NCUT'])
    
    ax1 = axes[i]
    ax2 = ax1.twinx()

    q_line, = ax1.plot(x, y_q, color='orange', marker='o', label='Q')
    ncut_line, = ax2.plot(x, y_ncut, color='green', marker='^', linestyle='--', label='NCUT')

    ax1.set_title(f"{method.capitalize()} Clustering")
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Modularity Q')
    ax2.set_ylabel('NCUT')

    ax1.tick_params(axis='y', colors='black')
    ax2.tick_params(axis='y', colors='black')

    # Combine legends from both axes into one
    ax1.legend(handles=[q_line, ncut_line], loc='best')

    ax1.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
