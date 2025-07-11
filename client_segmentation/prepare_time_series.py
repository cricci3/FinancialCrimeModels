import pandas as pd
import numpy as np

def prepare_timeseries_for_glasso(df):
    """
    Edit the p x n time series for glasso

    1. Compute the differences in the data to make it stationary.
    2. Divide each series by its std dev.

    Args:
        df (pd.DataFrame): A p x n dataframe where p (bank accounts) and n (account balance)
    Returns:
        pd.DataFrame: A processed p x (n-1) DataFrame ready for GLasso methods.
    """
    # Check if float
    Y = df.astype(float)

    # --- First-differencing to get stationary data
    # Compute the difference across time (columns, axis=1)
    Y_changes = Y.diff(axis=1)

    # First column will be NaN
    Y_changes = Y_changes.dropna(axis=1)

    # Standardization of the differenced data
    # Get the numpy array
    Y_matrix = Y_changes.values

    # Compute std dev per row of the differenced data
    row_std = np.std(Y_matrix, axis=1, keepdims=True)

    # No division by zero
    row_std[row_std == 0] = 1.0

    # Normalize each row (time series of an account's *changes*)
    Y_prepared = Y_matrix / row_std

    # Convert back to a DataFrame with original users and updated days
    return pd.DataFrame(Y_prepared, index=Y_changes.index, columns=Y_changes.columns)

# Example
data = {
    'Day1': [100, 5000],
    'Day2': [105, 4980],
    'Day3': [102, 5010],
    'Day4': [110, 5015]
}
df_original = pd.DataFrame(data, index=['UserA', 'UserB'])
df_prepared = prepare_timeseries_for_glasso(df_original)

print("Original DataFrame (Balances):")
print(df_original)
print("\nPrepared DataFrame (Standardized Daily Changes):")
print(df_prepared)

# Now you can pass df_prepared.values to SQUIC
# I'm showing it below for graphical lasso, but it's the same for SQUIC.
# from sklearn.covariance import GraphicalLasso
# glasso = GraphicalLasso(alpha=0.1)
# glasso.fit(df_prepared.T) # scikit-learn expects n x p
# precision_matrix = glasso.precision_