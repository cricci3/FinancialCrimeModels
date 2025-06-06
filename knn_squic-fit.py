from workflow.internal import load_dataset, knn_graph, normalization, visualize_metrics, extract_timeseries
from workflow.SQUIC_functions import squic_fit_matrix_computation, squic_fit_computation
from workflow.clustering_functions import clustering_2_communities, modularity_fscore, ARI_fscore, plot_ARI_f1, labels_to_partition, partition_to_labels
import matplotlib.pyplot as plt
import networkx as nx
import json
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import pandas as pd


if __name__ == '__main__':
    user_input = input("Do you want to load data? (Y/N)")
    user_input = user_input.upper()

    if user_input == 'Y':
        dimension = input("Which dimension?")
        dimension = dimension.upper()

        path = 'paysim_data_saved'

        try:
            # Set the chunk size (e.g., 100,000 rows per chunk)
            chunk_size = 1000

            chunks = []

            # Read CSV in chunks
            for chunk in pd.read_csv(f'{path}/YNorm_{dimension}.csv', chunksize=chunk_size):
                chunks.append(chunk)

            # Concatenate all chunks into a single DataFrame
            Y_norm = pd.concat(chunks, ignore_index=True)

            knn_matrix = sparse.load_npz(f'{path}/knn_matrix_{dimension}.npz')

            print(f"Shape of Y_norm loaded: {Y_norm.shape}")
            print(f"Shape of knn_matrix loaded: {knn_matrix.shape}")
            print(f"Type of knn_matrix loaded: {type(knn_matrix)}")

            with open(f'{path}/account_prop_{dimension}.json', 'r') as f:
                account_prop = json.load(f)

            name = 'PAYSIM'
        except:
            print(f"No saved data found for dimension {dimension}")
        
    else:
        # Load dataset (the user will pass the name)
        Y, name, dimension, account_prop, _ = load_dataset()

        # Extract time series
        # extract_timeseries(Y, name)

        # Normalise time series
        Y_norm = normalization(Y, name)

        print(Y_norm.shape)

        n_neighbors = 3
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', n_jobs=-1)
        nbrs.fit(Y_norm)
        knn_matrix = nbrs.kneighbors_graph(Y_norm, mode='connectivity') # sparse matrix

        user_input = input("Do you want to save this data? (Y/N)")
        user_input = user_input.upper()

        if user_input == 'Y':
            # Save to fast init
            path = 'paysim_data_saved'

            # Save Y norm to csv
            # Y_norm is rows = users, col = days
            df_save = pd.DataFrame(Y_norm)
            df_save.to_csv(f'{path}/YNorm_{dimension}.csv', index=False)

            # Save account prop as JSON
            with open(f'{path}/account_prop_{dimension}.json', 'w') as f:
                json.dump(account_prop, f)

            # Save knn matrix
            sparse.save_npz(f'{path}/knn_matrix_{dimension}.npz', knn_matrix)


    # Print knn matrix
    plt.figure(figsize=(7, 7))
    plt.spy(knn_matrix, markersize=5)
    plt.ylabel("Users", fontsize=18)

    plt.tick_params(axis='x')
    plt.tick_params(axis='y') 
    plt.show()

    results_squic = {}

    # Run SQUIC_fit
    results_squic['squic-fit-matrix'] = squic_fit_matrix_computation(Y_norm, name, dimension, knn_matrix, printMatrix=False)
    # results_squic['squic-fit-matrix'] = squic_fit_computation(Y_norm, name, dimension, printMatrix=True)

    dict_cluster = clustering_2_communities(results_squic, method='implemented')

    metrics = ARI_fscore(dict_cluster, results_squic, account_prop)

    for _, data in metrics.items():
        for rho, results in data.items():
            print(f"for rho = {rho} : {results}")

    plot_ARI_f1(metrics)
    