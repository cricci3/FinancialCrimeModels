import squic
import numpy as np
import scipy as sp
import time
from scipy.sparse import csr_matrix, lil_matrix, triu, find, isspmatrix_csr
from scipy.sparse import coo_matrix
import json 
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
import os
from functions.clustering_functions import study_CC
from functions.plots import print_covariance_matrix


'''
The following lib contains
- compute_squic -> function to compute SQUIC
- compute_squic_matrix -> function to compute SQUIC given a matrix as bias
- squic_fit_sparse -> function to compute SQUIC_Fit (sparse result)
- squic_fit_matrix_sparse -> function to compute SQUIC_Fit (sparse result) given a matrix as bias

Compute SQUIC/fit
- squic_computation
- squic_matrix_computation
- squic_fit_computation
- squic_fit_matrix_computation

Utility functions:
- nnz_sparse -> count number of nnz and nnz/row given a sparse matrix (SQUIC/Fit results)
- check_symmetric_sparse -> check if result of SQUIC/Fit is symmetric
- print_matrix -> print SQUIC/Fit result
'''

def read_lambdas(name, dimension, type):
    with open('lambda_values.json') as f:
            lambda_data = json.load(f)
        
    if name == 'LIBRA':
        lambdas = lambda_data[name]
    else:
        lambdas = lambda_data[name][dimension][type]
    
    return lambdas


def compute_squic(Y, lambda_val):
    '''
    Function to compute SQUIC

    Return:
    - X: Precision Matrix
    - time: computation time
    '''
    [X,_,times,_,_,_] = squic.run(Y,lambda_val)

    time = times[0]

    return X, time


def compute_squic_matrix(Y, lambda_val, bias_matrix):
    '''
    Function to compute SQUIC with matrix as bias

    Return:
    - X: Precision Matrix
    - Theta: inverse of X -> Covariance Matrix
    - time: computation time
    '''

    M_sparse = csr_matrix(bias_matrix)

    [X,_,times,_,_,_] = squic.run(Y,lambda_val, M=M_sparse)

    time = times[0]

    return X, time


def squic_fit_sparse(Y, lambda_val, eta):
    '''
    Sparse implementation of SQUIC-Fit
    
    Parameters:
    - Y: Input data matrix (sparse or dense)
    - lambda_val: Primary regularization parameter
    - eta: Secondary regularization parameter for biased entries
    - kappa: Threshold for identifying negative entries
    - tau: Threshold for final selection
    
    Returns:
    - X_final: Sparse adjacency matrix in CSR format
    - end_time: Computation time in seconds
    '''
    start_time = time.time()
    
    X1, _, _, _, _, _ = squic.run(Y=Y, l=lambda_val)
    X1 = X1.tocsr() if not isspmatrix_csr(X1) else X1

    kappa=0
    
    # Find negative off-diagonal elements < -kappa (upper triangle only)
    rows, cols, _ = find(triu(X1 < -kappa, k=1))
    
    # Create symmetric sparse matrix for G
    data = np.ones_like(rows)
    G = coo_matrix((np.concatenate([data, data]),
                       (np.concatenate([rows, cols]),
                        np.concatenate([cols, rows]))),
                      shape=X1.shape).tocsr()
    
    # Create sparse Lambda matrix with eta where G is nonzero
    Lambda_matrix = lil_matrix(X1.shape)
    Lambda_matrix[G.nonzero()] = eta
    Lambda_matrix = Lambda_matrix.tocsr()
    
    X2, _, _, _, _, _ = squic.run(Y=Y, l=lambda_val, M=Lambda_matrix)
    X2 = X2.tocsr() if not isspmatrix_csr(X2) else X2

    tau=0
    # Find negative off-diagonal elements < -tau
    rows, cols, vals = find(triu(X2 < -tau, k=1))
    
    # Create symmetric COO matrix directly
    data = X2[rows, cols].A1  # Extract values as 1D array
    X_final = coo_matrix((np.concatenate([data, data]),
                       (np.concatenate([rows, cols]),
                        np.concatenate([cols, rows]))),
                      shape=X2.shape).tocsr()
    
    end_time = round(time.time() - start_time, 2)
    return X_final, end_time


def squic_fit_matrix_sparse(Y, l, bias_matrix, eta=0):
    '''
    Sparse implementation of SQUIC Fit with bias
    [...] We define Λij = Mij if Mij!=0
                and Λij = λ, if otherwise.
    
    SQUIC Fit algorithm but we skip first SQUIC call

    Return:
    - X_final: adjacency matrix
    - end_time: computation time
    '''
    start_time = time.time()

    # Construct Lambda matrix (bias)
    if eta == 0:
        L_matrix = csr_matrix(bias_matrix)
    else: # Favor edges from the KNN graph
        # Λij = η if KNN says i∼j
        # Λij = λ otherwise
        L_matrix = csr_matrix(bias_matrix) * eta

    # Run SQUIC with bias
    X2, _, _, _, _, _ = squic.run(Y=Y, l=l, M=L_matrix)

    X2.setdiag(0)
    
    # Ensure is spare
    X2 = X2.tocsr() if not isspmatrix_csr(X2) else X2

    # Find negative off-diagonal elements < -tau (only upper triangle to avoid duplicates)
    rows, cols, vals = find(triu(X2, k=1))  # k=1 excludes diagonal

    # Construct X_final
    X_final = lil_matrix(X2.shape, dtype=X2.dtype) # I want to save X2 into a sparse matrix, so create nxn sparse matrix

    # Set values in X_final
    for i, j, val in zip(rows, cols, vals):
        if val < 0: # Store absolute value of negative entry
            abs_val = abs(val)
            X_final[i, j] = abs_val
            X_final[j, i] = abs_val  # symmetric

    # Convert back to CSR
    X_final = X_final.tocsr()
    
    end_time = round(time.time() - start_time, 2)
    return X_final, end_time
    

def nnz_sparse(X, rows):
    '''
    Function to count nnz on a matrix
    
    Returns:
    - nnz: number of non-zero elements
    - nnz_r: number of non-zero elements per row
    '''
    A = sp.sparse.csr_array(X)
    nnz = A.count_nonzero()
    nnz_r = nnz / rows
    return nnz, round(nnz_r, 2)


def check_symmetric_sparse(X):
    if X.shape[0] != X.shape[1]:
        return False
    
    # Check if the sparsity pattern is symmetric first (quick check)
    if not (X != X.T).nnz == 0:
        return False
    
    # If pattern is symmetric, check values
    difference = X - X.T
    if difference.nnz == 0:
        return True
    else:
        return False


def squic_computation(Y_norm, name, dimension, printMatrix=False, save=False, path=None, lambdas="no-bias"):
    lambdas = read_lambdas(name, dimension, lambdas)

    ROWS = len(Y_norm)

    squic_method = 'squic'

    # Dict to store precision matrix given by SQUIC
    theta_dict = {}

    for rho in lambdas:
        theta_dict[rho], end_time = compute_squic(Y_norm, lambda_val=rho)

        end_time = round(end_time, 2)
        print(f"required time: {end_time}")

        nnz, nnz_r = nnz_sparse(theta_dict[rho], ROWS)
        print(f"nnz = {nnz} per rows = {nnz_r}")

        if printMatrix or save:
            print_covariance_matrix(theta_dict[rho], printMatrix, save, path=f'images/{name}/{dimension}/', file_name=f'{squic_method}_{str(rho).replace(".","_")}')

        if check_symmetric_sparse(theta_dict[rho]):
            print(f" Matrix is symmetric per rho {rho}")
        else:
            print(f" Matrix is not symmetric per rho {rho}")

    return theta_dict


def squic_matrix_computation(Y_norm, name, dimension, adjaceny_matrix, printMatrix=False, save=False, path=None):
    lambdas = read_lambdas(name, dimension, "bias")

    ROWS = len(Y_norm)

    squic_method = 'squic-matrix'

    # Dict to store precision matrix given by SQUIC
    theta_dict = {}

    for rho in lambdas:
        theta_dict[rho], end_time = compute_squic_matrix(Y_norm, lambda_val=rho, bias_matrix=adjaceny_matrix)
        end_time = round(end_time, 2)
        print(f"required time: {end_time}")

        nnz, nnz_r = nnz_sparse(theta_dict[rho], ROWS)
        print(f"nnz = {nnz} per rows = {nnz_r}")

        if printMatrix or save:
            print_covariance_matrix(theta_dict[rho], printMatrix, save, path=f'images/{name}/{dimension}/', file_name=f'{squic_method}_{str(rho).replace(".","_")}')

        if check_symmetric_sparse(theta_dict[rho]):
            print(f" Matrix is symmetric per rho {rho}")
        else:
            print(f" Matrix is not symmetric per rho {rho}")

    return theta_dict


def squic_fit_computation(Y_norm, name, dimension, printMatrix=False, save=False, path=None):
    
    lambdas = read_lambdas(name, dimension, "no-bias")

    ROWS = len(Y_norm)

    squic_method = 'squic-fit'

    W_matrices = {}
    times_dict = {}

    for rho in lambdas:
        W_matrices[rho], end_time = squic_fit_sparse(Y_norm, rho, rho/10)
        end_time = round(end_time, 2)
        print(f"required time: {end_time}")
        times_dict[rho] = end_time

        # Get diagonal elements
        diagonal = W_matrices[rho].diagonal()

        # Check if all diagonal element are zero
        if not np.all(diagonal == 0):
            print("There are some non zero(s) on the diagonal.")

        nnz, nnz_r = nnz_sparse(W_matrices[rho], ROWS)
        print(f"nnz = {nnz} per rows = {nnz_r}")

        if printMatrix or save:
            print_covariance_matrix(W_matrices[rho], printMatrix, save, path=f'images/{name}/{dimension}/', file_name=f'{squic_method}_{str(rho).replace(".","_")}')

        if check_symmetric_sparse(W_matrices[rho]):
            print(f" Matrix is symmetric per rho {rho}")
        else:
            print(f" Matrix is not symmetric per rho {rho}")

    return W_matrices, times_dict


def squic_fit_matrix_computation(Y_norm, name, dimension, adjaceny_matrix, study=False, printMatrix=False, save=False):
    
    lambdas = read_lambdas(name, dimension, "bias")

    squic_method = 'squic-fit-matrix'

    ROWS = len(Y_norm)

    W_matrices = {}
    times_dict = {}

    for rho in lambdas:
        #W_matrices[rho], end_time = squic_fit_matrix_sparse(Y=Y_norm, l=rho, bias_matrix=adjaceny_matrix, eta=rho/10)
        W_matrices[rho], end_time = squic_fit_matrix_sparse(Y=Y_norm, l=rho, bias_matrix=adjaceny_matrix, eta=0)
        end_time = round(end_time, 2)
        print(f"required time: {end_time}")

        times_dict[rho] = end_time
        # Get diagonal elements
        diagonal = W_matrices[rho].diagonal()

        # Check if all diagonal element are zero
        if not np.all(diagonal == 0):
            print("There are some non zero(s) on the diagonal.")

        nnz, nnz_r = nnz_sparse(W_matrices[rho], ROWS)
        print(f"nnz = {nnz} per rows = {nnz_r}")

        if printMatrix or save:
            print_covariance_matrix(W_matrices[rho], printMatrix, save, path=f'images/{name}/{dimension}/', file_name=f'{squic_method}_{str(rho).replace(".","")}')

        if check_symmetric_sparse(W_matrices[rho]):
            print(f" Matrix is symmetric per rho {rho}")
        else:
            print(f" Matrix is not symmetric per rho {rho}")

        if study:
            study_CC(W_matrices[rho])

    return W_matrices, times_dict
