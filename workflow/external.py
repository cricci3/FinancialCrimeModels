import json
from workflow.preprocess import PaySim_preprocessing
from workflow.SQUIC_functions import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score


def parse_input(user_input):
    """Parse and validate the dataset input in the format NAME_DIMENSION."""
    try:
        name, dimension = user_input.strip().upper().split("_")
    except ValueError:
        raise ValueError("Input must be in the format NAME_DIMENSION (e.g., PAYSIM_10K)")

    valid_names = {"PAYSIM"}
    valid_dimensions = {"100", "1K", "10K", "100K", "1M"}

    if name not in valid_names:
        raise ValueError(f"Invalid dataset name '{name}'. Valid options are: {', '.join(valid_names)}")
    if dimension not in valid_dimensions:
        raise ValueError(f"Invalid dimension '{dimension}'. Valid options are: {', '.join(valid_dimensions)}")

    return name, dimension

def load_dataset():
    """Prompt user input and load the corresponding dataset."""
    while True:
        user_input = input("Insert dataset name in the following format NAME_DIMENSION (e.g., AMLSIM_10K): ")

        try:
            name, dimension = parse_input(user_input)
            break
        except ValueError as e:
            print(f"Error: {e}")
            continue

    if name == 'PAYSIM':
        df = PaySim_preprocessing(dimension)

    return df, name, dimension


def squic_computation(Y_norm, name, dimension, printMatrix=False):
    with open('lambda_values.json') as f:
            lambda_data = json.load(f)
        
    lambdas = lambda_data[name][dimension]["norm"]

    ROWS = len(Y_norm)

    # Dict to store covariance matrix given by SQUIC
    theta_dict = {}

    data_nnz = []
    data_nnzr = []
    data_time = []
    data_sym = []

    for rho in lambdas:
        _, theta_dict[rho], end_time = compute_squic(Y_norm, lambda_val=rho)
        end_time = round(end_time, 2)
        print(f"required time: {end_time}")

        nnz, nnz_r = nnz_fit(theta_dict[rho], ROWS)
        print(f"nnz = {nnz} per rows = {nnz_r}")

        if printMatrix:
            sparsity_pattern(theta_dict[rho])

        if is_symmetric(theta_dict[rho]):
            #print(f"✅ Matrix is symmetric per rho {rho}")
            data_sym.append("Yes")
        else:
            print(f"❌ Matrix is not symmetric per rho {rho}")
            data_sym.append("No")

        data_nnz.append(nnz)
        data_nnzr.append(nnz_r)
        data_time.append(end_time)

    table_norm = [
            ["NNZ"] + data_nnz,
            ["NNZ/Row"] + data_nnzr,
            ["Time (s)"] + data_time,
            ["Symmetric"] + data_sym
    ]

    return theta_dict, table_norm


def lda(Theta_matrices, df):
    X_df = df.T

    y = X_df.index.str.startswith('C').astype(int)  # 1 for C‑columns, 0 for M‑columns

    # Class means (p×1)
    mu_M = X_df[y == 0].mean(axis=0).to_numpy()
    mu_C = X_df[y == 1].mean(axis=0).to_numpy()
    mean_diff = (mu_C - mu_M).reshape(-1, 1)
    
    dict_scores = {}

    for l, Theta in Theta_matrices.items():
        print(Theta.shape)
        # LDA weight vector
        w = Theta @ mean_diff
        w /= np.linalg.norm(w)

        # Project samples
        scores = X_df.values @ w

        dict_scores[l] = scores
    
    return dict_scores, y


def external_metrics(dict_scores, y):
    ext_metrics_dict = {}

    for l, score in dict_scores.items():
        threshold = 0.0  # classic LDA uses 0; tune on validation if needed
        y_pred = (score > threshold).astype(int)

        f1 = f1_score(y, y_pred, pos_label=1)
        ext_metrics_dict[l] = f1

    return ext_metrics_dict

        

