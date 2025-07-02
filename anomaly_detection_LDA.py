from functions.internal import extract_timeseries, normalization
from functions.external import load_dataset, prepare_LDA , external_metrics, visualize_graph_external
from functions.SQUIC_functions import squic_computation
import numpy as np
import pandas as pd


def show_df(Y):
    n_users, n_days = Y.shape

    # Create labels
    day_labels = [f"Day {i}" for i in range(n_days)]
    user_labels = list(range(n_users))

    # Create the DataFrame
    df = pd.DataFrame(Y, index=user_labels, columns=day_labels)

    print(df)


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    Y, name, dimension, account_prop = load_dataset()

    # Extract time series
    extract_timeseries(Y, name, dimension)

    Y = Y.T.values  # Convert to (users, days)

    show_df(Y)

    # Normalise time series
    Y_norm = normalization(Y, name)
    extract_timeseries(Y_norm, name, dimension, type_df='norm')

    show_df(Y_norm)

    std_dev = round(np.mean(np.std(Y_norm, axis=1)), 2)

    print("Mean std per row (should be 1):", std_dev)
    print("Max:", np.max(Y_norm))
    print("Min:", np.min(Y_norm))

    # Run SQUIC to compute Θ
    Theta_matrices = squic_computation(Y_norm, name, dimension)

    # extract Θ for LDA
    ext_scores = prepare_LDA(Theta_matrices, account_prop, target='fraud')

    # Report external metrics on the classification of Θ
    ext_metrics = external_metrics(ext_scores)
    #print(ext_metrics)

    # Visualise with cosmograph Θ
    # visualize_graph_external(Theta_matrices, account_prop, name, dimension)
