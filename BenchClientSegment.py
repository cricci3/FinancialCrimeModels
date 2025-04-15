from preprocessing.preprocess import *
import matplotlib.pyplot as plt
import json
import numpy as np
from squic.SQUIC_functions import *


def load_dataset():
    dataset = input("Insert dataset name in the following format NAME_DIMENSION : ")

    dataset = dataset.split("_")

    name = dataset[0]
    dimension = dataset[1]

    name = name.upper()

    valid_dimensions = ["100", "1K", "10K", "100K", "1M"]

    if dimension in valid_dimensions:
        if name == 'AMLSIM':
            df = AMLSim_preprocessing(dimension)
        elif name == 'PAYSIM':
            df = PaySim_preprocessing(dimension)
        elif name == 'LIBRE':
            pass
        else:
            print("Invalid dataset name")
    else:
        print("Invalid dataset dimension")

    return df, name, dimension

def extract_timeseries(df):
    # Plot the balance evolution for all users (columns) as separate lines
    # plt.figure(figsize=(15,10), dpi= 300)
    plt.figure(figsize=(15,10))

    # Plot each column (account balance) as a line
    for user in df.columns:
        # color = 'mediumseagreen' if user.startswith('C') else 'hotpink'

        plt.plot(df.index, df[user], label=f"User {user}", alpha=0.6)

    # Add title and labels
    plt.xlabel("Days", fontsize=22)
    plt.ylabel("Balance", fontsize=22)

    # Rotate x-axis labels for better visibility
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)

    # Display the plot
    plt.tight_layout()
    plt.show()
    

def normalization(df, name):
    Y = df.T.to_numpy()  # Convert to (users, days)

    if name == 'AMLSIM':
        # normalize the Y input data to achieve unit variance
        Y_new = np.diag(1/np.std(Y,1)) @ Y # each row of Y is scaled by the inverse of its standard deviation
    elif name == 'PAYSIM':
        stds = np.std(Y, axis=1)
        safe_stds = np.clip(stds, 1e-8, None)  # don't allow std < 1e-8
        Y_new = np.diag(1 / safe_stds) @ Y

    return Y_new


def compute_squic(Y_norm, name, dimension):
    with open('lambda_values.json') as f:
            lambda_data = json.load(f)
        
    lambdas = lambda_data[name][dimension]["norm"]

    ROWS = len(Y_norm)

    fit_norm_dict = {}

    data_nnz = []
    data_nnzr = []
    data_time = []
    data_sym = []

    if name == 'AMLSIM':
        for rho in lambdas:
            fit_norm_dict[rho], end_time = squic_fit(Y_norm, lambda_val=rho, eta=rho * 0.1)
            end_time = round(end_time, 2)
            print(f"required time: {end_time}")

            nnz, nnz_r = nnz_fit(fit_norm_dict[rho], ROWS)
            print(f"nnz = {nnz} per rows = {nnz_r}")

            sparsity_pattern(fit_norm_dict[rho])

            if is_symmetric(fit_norm_dict[rho]):
                print(f"✅ Matrix is symmetric per rho {rho}")
                data_sym.append("Yes")
            else:
                print(f"❌ Matrix is not symmetric per rho {rho}")
                data_sym.append("No")

            data_nnz.append(nnz)
            data_nnzr.append(nnz_r)
            data_time.append(end_time)

        table_fit_norm = [
            ["NNZ"] + data_nnz,
            ["NNZ/Row"] + data_nnzr,
            ["Time (s)"] + data_time,
            ["Symmetric"] + data_sym
        ]

    elif name == 'PAYSIM':
        pass
    else:
        # libra
        pass

    return fit_norm_dict, table_fit_norm



def clustering(X):
    pass

def internal_metrics(X):
    pass

def create_graph(X):
    pass


if __name__ == '__main__':
    # Load dataset (the user will pass the name)
    df, name, dimension = load_dataset()

    # Extract time series
    extract_timeseries(df)

    # Normalise time series
    Y_norm = normalization(df)

    # Run SQUIC_fit
    squic_results, table_results = compute_squic(Y_norm, name, dimension)
    
    # # Use the extracted W (check if symmetric) for clustering
    # clustering()

    # # Report internal metrics on the clustering, and visualise with cosmograph
    # internal_metrics()
    # create_graph()