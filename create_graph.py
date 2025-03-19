import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering, DBSCAN

def create_graph(matrix):
    # Ensure all off-diagonal entries are positive
    theta2 = np.abs(matrix) 

    # Ensure diagonal entries are zero
    np.fill_diagonal(matrix, 0)

    G = nx.from_numpy_array(matrix)

    cluster_range = range(2, 11)
    sil_scores = []

    for n in cluster_range:
        # Perform SpectralClustering
        clustering = SpectralClustering(n_clusters=n, affinity='precomputed', assign_labels='cluster_qr')
        labels = clustering.fit_predict(np.asarray(matrix))
        
        # Compute the Silhouette Score
        sil_score = silhouette_score(np.asarray(matrix), labels, metric='precomputed')
        sil_scores.append(sil_score)

    # Plot Silhouette Score for each number of clusters
    plt.plot(cluster_range, sil_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method')
    plt.show()

    n_clusters = 5
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='cluster_qr')
    labels = clustering.fit_predict(np.asarray(matrix))

    # Convert the labels to a DataFrame (node indices and their cluster labels)
    df_labels = pd.DataFrame({
        'Node': range(len(labels)),
        'Cluster': labels
    })

    df_labels.to_csv('clustered_nodes.csv', index=False)