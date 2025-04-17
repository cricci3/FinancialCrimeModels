from sklearn.cluster import DBSCAN
from networkx.algorithms import community


for rho in l:
    X = fit_norm_dict[rho]

    # Ensure all off-diagonal entries are positive
    X = np.abs(X) 

    # Ensure diagonal entries are zero
    np.fill_diagonal(X, 0)

    # Create a graph from matrix X
    G = nx.from_numpy_array(X)

    # DBSCAN
    dbscan = DBSCAN(eps=0.2, min_samples=2)
    labels_db = dbscan.fit_predict(np.asarray(X))

    # # the number of clusters found by DBSCAN
    # n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    # print(f"Number of clusters found by DBSCAN: {n_clusters} for rho {rho}")

    # SpectraclClustering
    clustering = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='cluster_qr')
    labels_spec = clustering.fit_predict(np.asarray(X))

    # Louvain
    partition = community.louvain_communities(G) # set of nodes forming a cluster
                                                # [{4} {0, 1}] ex node 4 in cluster 1, node 0 and 1 in cluster 2
    # Convert to node -> cluster index
    node_to_community = {}
    for idx, comm in enumerate(partition):
        for node in comm:
            node_to_community[node] = idx

    labels_louvain = [node_to_community[n] for n in range(len(node_to_community))] # list with labels associated at each node

    # pos = nx.spring_layout(G, seed=42)
    # colors = [node_to_community[n] for n in G.nodes()]

    # plt.figure(figsize=(12, 8))
    # nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.tab10, node_size=100)
    # nx.draw_networkx_edges(G, pos, alpha=0.2)
    # plt.title(f"Louvain Clustering of Precision Matrix Graph (rho = {rho})")
    # plt.axis('off')
    # plt.show()

    # Modularity
    # modularity_db = community.modularity(G, [{i for i, x in enumerate(labels_db) if x == c} for c in np.unique(labels_db)])
    
    # modularity_spec = community.modularity(G, [{i for i, x in enumerate(labels_spec) if x == c} for c in np.unique(labels_spec)])

    modularity_louvain = community.modularity(G, partition)
    # print(f"modularity DBSCAN for rho {rho} is {float(round(modularity_db, 2))}")
    # print(f"modularity SPECTRAL for rho {rho} is {float(round(modularity_spec, 2))}")
    print(f"modularity LOUVAIN for rho {rho} is {float(round(modularity_louvain, 2))}")

    # n_cut = 0
    # r_cut = 0

    # unique_labels_db = np.unique(labels_db)
    # unique_labels_sp = np.unique(labels_spec)
    unique_labels_louv = np.unique(labels_louvain)

        
    # for cluster in unique_labels_db:
    #     mask = (labels_db == cluster)
    #     not_mask = ~mask
    #     cut = X[mask][:, not_mask].sum()
    #     vol = X[mask].sum()
    #     assoc = X[mask][:, mask].sum()
        
    #     n_cut += cut / (vol + 1e-10)  # Avoid division by zero
    #     r_cut += cut / (mask.sum() + 1e-10)  # Normalize by cluster size

    # print(f"n cut for DBSCAN is: {float(round(n_cut, 2))}")
    # print(f"r cut for DBSCAN is: {float(round(r_cut, 2))}")

    # n_cut = 0
    # r_cut = 0

    # for cluster in unique_labels_sp:
    #     mask = (labels_spec == cluster)
    #     not_mask = ~mask
    #     cut = X[mask][:, not_mask].sum()
    #     vol = X[mask].sum()
    #     assoc = X[mask][:, mask].sum()
        
    #     n_cut += cut / (vol + 1e-10)  # Avoid division by zero
    #     r_cut += cut / (mask.sum() + 1e-10)  # Normalize by cluster size

    # print(f"n cut for SPEC is: {float(round(n_cut, 2))}")
    # print(f"r cut for SPEC is: {float(round(r_cut, 2))}")

    n_cut = 0
    r_cut = 0

    for cluster in unique_labels_louv:
        mask = (labels_louvain == cluster)
        not_mask = ~mask
        cut = X[mask][:, not_mask].sum()
        vol = X[mask].sum()
        assoc = X[mask][:, mask].sum()
        
        n_cut += cut / (vol + 1e-10)  # Avoid division by zero
        r_cut += cut / (mask.sum() + 1e-10)  # Normalize by cluster size

    print(f"n cut for LOUV is: {float(round(n_cut, 2))}")
    print(f"r cut for LOUV is: {float(round(r_cut, 2))}")
    
    print('\n')

cc = nx.number_connected_components(G)
if not G.is_directed():
    G_dir = G.to_directed()
else:
    G_dir = G
scc = nx.number_strongly_connected_components(G_dir)

print(f"CC = {cc}")
print(f"SCC = {scc}")   