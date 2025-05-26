import matplotlib.pyplot as plt
import squic
import scipy as sp
import numpy as np
import time
from scipy.sparse import csr_matrix, lil_matrix, triu, find, isspmatrix_csr


'''
The following lib contains
- compute_squic -> function to compute SQUIC
- compute_squic_matrix -> function to compute SQUIC given a matrix as bias
- squic_fit_sparse -> function to compute SQUIC_Fit (sparse result)
- squic_fit_matrix_sparse -> function to compute SQUIC_Fit (sparse result) given a matrix as bias

Utility functions:
- nnz_sparse -> count number of nnz and nnz/row given a sparse matrix (SQUIC/Fit results)
- check_symmetric_sparse -> check if result of SQUIC/Fit is symmetric
- print_matrix -> print SQUIC/Fit result
'''


def compute_squic(Y, lambda_val):
    '''
    Function to compute SQUIC

    Return:
    - X: Precision Matrix
    - Theta: inverse of X -> Covariance Matrix
    - time: computation time
    '''
    [X,Theta,times,_,_,_] = squic.run(Y,lambda_val)

    time = times[0]

    return X, Theta, time


def compute_squic_matrix(Y, lambda_val, bias_matrix):
    '''
    Function to compute SQUIC with matrix as bias

    Return:
    - X: Precision Matrix
    - Theta: inverse of X -> Covariance Matrix
    - time: computation time
    '''

    M_sparse = csr_matrix(bias_matrix)

    [X,Theta,times,_,_,_] = squic.run(Y,lambda_val, M=M_sparse)

    time = times[0]

    return X, Theta, time


def squic_fit_sparse(Y, lambda_val, eta, kappa=0, tau=0):
    '''
    Sparse implementation of SQUIC-Fit

    Return:
    - X_final: adjacency matrix
    - end_time: computation time
    '''
    start_time = time.time()
    # First squic call -> Identify negative off-diagonal elements (Equation 9)
    X1, _, _, _, _, _  = squic.run(Y, lambda_val)
    X1 = X1.todense()
    
    # Step 2: Build Graphical Bias G (Equation 10)
    G = np.zeros_like(X1)
    G[np.triu_indices_from(G, k=1)] = (X1[np.triu_indices_from(X1, k=1)] < -kappa).astype(int)
    G += G.T  # Make symmetric
    
    # Step 3: Build Regularization Parameter Matrix Λ (Equation 12)
    #Lambda = np.full_like(Theta1, lambda_val) 
    Lambda = np.zeros_like(X1)
    # Apply eta where G is nonzero
    Lambda[G != 0] = eta
    
    # Step 4: Second SQUIC estimation with bias (Equation 11)
    X2, _, _, _, _, _ = squic.run(Y, lambda_val, M=Lambda)
    X2 = X2.todense()
    
    # Step 5: Construct the final M-matrix (Equation 13)
    X_final = np.zeros_like(X2)
    
    # Get diagonal
    diag = np.diag(X2)
    
    # Identify negative off-diagonal elements
    n = X2.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if X2[i, j] < -tau:
                X_final[i, j] = X2[i, j]
                X_final[j, i] = X2[j, i]
    
    # Restore diagonal
    np.fill_diagonal(X_final, diag)
    end_time = time.time() - start_time
    end_time = round(end_time, 2)
    
    return X_final, end_time


def squic_fit_matrix_sparse(Y, l, bias_matrix, tau=0):
    '''
    Sparse implementation of function to pass a matrix as bias to SQUIC_Fit

    Return:
    - X_final: adjacency matrix
    - end_time: computation time
    '''
    start_time = time.time()

    M_sparse = csr_matrix(bias_matrix)

    # Step 4: Run SQUIC
    X2, _, _, _, _, _ = squic.run(Y, l, M=M_sparse)
    
    X2 = X2.tocsr() if not isspmatrix_csr(X2) else X2

    # Step 5: Construct X_final
    n = X2.shape[0]
    X_final = lil_matrix((n, n), dtype=X2.dtype)

    # Copy the diagonal
    # diag = X2.diagonal()
    # X_final.setdiag(diag)

    # Find negative off-diagonal elements < -tau (only upper triangle to avoid duplicates)
    rows, cols, vals = find(triu(X2 < -tau, k=1))  # k=1 excludes diagonal

    # Set values in X_final (symmetric update)
    for i, j, val in zip(rows, cols, vals):
        X_final[i, j] = X2[i, j]
        X_final[j, i] = X2[j, i]

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


def print_matrix(X, save=False, path=None):
    '''
    Function to print adjaceny matrix
    '''
    #plt.figure(figsize=(10, 10), dpi=300)
    plt.figure(figsize=(7, 7))
    # plt.spy(X, markersize=5, c="#484154")
    plt.spy(X, markersize=5)
    # plt.xlabel("Users", fontsize=18)
    plt.ylabel("Users", fontsize=18)
    # plt.ylabel("Users")

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)  
    # plt.title("Sparsity Pattern of Precision Matrix (X)")

    if save:
        plt.savefig(path)
    
    plt.show()


# def squic_fit(Y, lambda_val, eta, kappa=0, tau=0):
#     '''
#     Dense implementation of SQUIC-Fit

#     Return:
#     - X_final: adjacency matrix
#     - end_time: computation time
#     '''
#     start_time = time.time()
#     # First squic call -> Identify negative off-diagonal elements (Equation 9)
#     X1, _, _, _, _, _  = squic.run(Y, lambda_val)
#     X1 = X1.todense()
    
#     # Step 2: Build Graphical Bias G (Equation 10)
#     G = np.zeros_like(X1)
#     G[np.triu_indices_from(G, k=1)] = (X1[np.triu_indices_from(X1, k=1)] < -kappa).astype(int)
#     G += G.T  # Make symmetric
    
#     # Step 3: Build Regularization Parameter Matrix Λ (Equation 12)
#     #Lambda = np.full_like(Theta1, lambda_val) 
#     Lambda = np.zeros_like(X1)
#     # Apply eta where G is nonzero
#     Lambda[G != 0] = eta
    
#     # Step 4: Second SQUIC estimation with bias (Equation 11)
#     X2, _, _, _, _, _ = squic.run(Y, lambda_val, M=Lambda)
#     X2 = X2.todense()
    
#     # Step 5: Construct the final M-matrix (Equation 13)
#     X_final = np.zeros_like(X2)
    
#     # Get diagonal
#     diag = np.diag(X2)
    
#     # Identify negative off-diagonal elements
#     n = X2.shape[0]
#     for i in range(n):
#         for j in range(i+1, n):
#             if X2[i, j] < -tau:
#                 X_final[i, j] = X2[i, j]
#                 X_final[j, i] = X2[j, i]
    
#     # Restore diagonal
#     np.fill_diagonal(X_final, diag)
#     end_time = time.time() - start_time
#     end_time = round(end_time, 2)
    
#     return X_final, end_time


# def squic_fit_matrix(Y, l, bias_matrix, tau=0):
#     '''
#     Dense implementation of function to pass a matrix as bias to SQUIC_Fit

#     Return:
#     - X_final: adjacency matrix
#     - end_time: computation time
#     '''
#     start_time = time.time()

#     M_sparse = csr_matrix(bias_matrix)

#     # Step 4: Second SQUIC estimation with bias (Equation 11)
#     X2, _, _, _, _, _ = squic.run(Y, l, M=M_sparse)
#     X2 = X2.todense()
    
#     # Step 5: Construct the final M-matrix (Equation 13)
#     X_final = np.zeros_like(X2)
    
#     # Get diagonal
#     diag = np.diag(X2)
    
#     # Identify negative off-diagonal elements
#     n = X2.shape[0]
#     for i in range(n):
#         for j in range(i+1, n):
#             if X2[i, j] < -tau:
#                 X_final[i, j] = X2[i, j]
#                 X_final[j, i] = X2[j, i]
    
#     # Restore diagonal
#     np.fill_diagonal(X_final, diag)
    
#     end_time = round(time.time() - start_time, 2)
    
#     return X_final, end_time


# def count_nnz(X, rows):
#     '''
#     Function to count nnz on a matrix
    
#     Returns:
#     - nnz: number of non-zero elements
#     - nnz_r: number of non-zero elements per row
#     '''
#     nnz = X.nnz
#     nnz_r = nnz / rows
#     return nnz, round(nnz_r, 2)


# def is_symmetric(X):
#     if sp.sparse.issparse(X):
#         X = X.toarray()
#     return np.allclose(X, X.T)