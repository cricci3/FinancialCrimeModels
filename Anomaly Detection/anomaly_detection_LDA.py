from functions.internal import extract_timeseries, normalization
from functions.external import load_dataset, prepare_LDA , external_metrics, visualize_graph_external
from functions.SQUIC_functions import squic_computation
import numpy as np
import pandas as pd
import os
import json


def ask_yes_no(prompt, default=None):
    """
    Ask the user a yes/no question with optional default.
    """
    while True:
        answer = input(f"{prompt} (Y/N) ").strip().upper()
        if not answer and default:
            return default
        if answer in ['Y', 'N']:
            return answer
        print("Please enter 'Y' or 'N'.")


def ask_input(prompt, default=None):
    """
    Ask the user for input with optional default.
    """
    answer = input(f"{prompt} ").strip()
    if not answer and default is not None:
        return default
    return answer


def show_df(Y):
    n_users, n_days = Y.shape

    # Create labels
    day_labels = [f"Day {i}" for i in range(n_days)]
    user_labels = list(range(n_users))

    # Create the DataFrame
    df = pd.DataFrame(Y, index=user_labels, columns=day_labels)

    print(df)


if __name__ == '__main__':
    user_input = ask_yes_no("\nDo you want to load data?")

    if user_input == 'Y':
        name = ask_input("Which dataset?").upper()
        dimension = ask_input("Which dimension (100/1K/10K/100K/1M)?").upper()
        path = f'{name}_data_saved'

        try:
            # Load pre-saved Y_norm
            chunk_size = 10000
            chunks = []

            for chunk in pd.read_csv(f'{path}/YNorm_{dimension}.csv', chunksize=chunk_size):
                chunks.append(chunk)

            Y_norm = pd.concat(chunks, ignore_index=True)
            print(f"Head of Y_norm df:")
            print(Y_norm.head())

            print(f"\ndimension of YNorm loaded: {Y_norm.shape}")

            std_dev = round(np.mean(np.std(Y_norm, axis=1)), 2)

            print("Mean std per row (should be 1):", std_dev)
            print("Max:", np.max(Y_norm))
            print("Min:", np.min(Y_norm))
            
            # Load pre-saved account_properties
            with open(f'{path}/account_prop_{dimension}.json', 'r') as f:
                account_prop = json.load(f)

            account_prop = {int(k): v for k, v in account_prop.items()}
            
        except Exception as e:
            # If data are not present
            print(f"Error loading data: {e}")
            exit(1)
        
    else: # Normal run
        # Load dataset (the user will pass the name)
        Y, name, dimension, account_prop = load_dataset()

        # Extract time series
        extract_timeseries(Y, name, dimension)

        Y = Y.T.values  # Convert to (users, days)

        show_df(Y)

        # Normalise time series
        Y_norm = normalization(Y, name)

        # extract_timeseries(Y_norm, name, dimension, type_df='norm')

        show_df(Y_norm)

        std_dev = round(np.mean(np.std(Y_norm, axis=1)), 2)

        print("Mean std per row (should be 1):", std_dev)
        print("Max:", np.max(Y_norm))
        print("Min:", np.min(Y_norm))

        # Ask user if want to save the data for next runs
        if ask_yes_no("Do you want to cache this data?") == 'Y':
            path = f'{name}_data_saved'

            # if path does not exists, create it
            os.makedirs(path, exist_ok=True)
            
            # Save Y_norm as CSV
            pd.DataFrame(Y_norm).to_csv(f'{path}/YNorm_{dimension}.csv', index=False)
            
            # Save account_prop as json
            with open(f'{path}/account_prop_{dimension}.json', 'w') as f:
                json.dump(account_prop, f)
    
    # Run SQUIC to compute Θ
    Theta_matrices = squic_computation(Y_norm, name, dimension, printMatrix=True)

    # extract Θ for LDA
    print("Enter LDA")
    ext_scores = prepare_LDA(Theta_matrices, account_prop, target='fraud')

    # Report external metrics on the classification of Θ
    ext_metrics = external_metrics(ext_scores)
    #print(ext_metrics)

    # Visualise with cosmograph Θ
    # visualize_graph_external(Theta_matrices, account_prop, name, dimension)
