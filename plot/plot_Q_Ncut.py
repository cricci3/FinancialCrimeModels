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
        'louvain': {'NCUT' : 2.64, 'Q' : 0.82, 'nCluster' : 388},
        'leiden': {'NCUT' : 5.02, 'Q' : 0.84, 'nCluster' : 407},
        'dbscan': {'NCUT' : 284.24, 'Q' : 0.22, 'nCluster' : 612},
        'spectral': {'NCUT' : 0.0, 'Q' : 0.2, 'nCluster' : 6}
    }
}    


import matplotlib.pyplot as plt

methods = ['louvain', 'leiden', 'dbscan', 'spectral']
colors = {'louvain': 'blue',
          'leiden': 'green',
          'dbscan': 'red',
          'spectral': 'purple'}
markers = {'100': 'o', '1K': 's', '10K' : 'D'}

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

for method in methods:
    x = []
    y_q = []
    y_ncut = []
    for size in ['100', '1K', '10K']:
        entry = results[size][method]
        x.append(entry['nCluster'])
        y_q.append(entry['Q'])
        y_ncut.append(entry['NCUT'])

    ax1.plot(x, y_q, color=colors[method], label=f'{method} (Q)', marker='o')
    ax2.plot(x, y_ncut, color=colors[method], linestyle='--', label=f'{method} (NCUT)', marker='x')

# Labels
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Modularity Q', color='black')
ax2.set_ylabel('NCUT', color='black')

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

# Title and layout
plt.grid(True)
plt.tight_layout()
plt.show()