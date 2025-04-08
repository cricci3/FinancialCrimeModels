import matplotlib.pyplot as plt
import squic
import scipy as sp
import numpy as np
import time
from scipy.sparse import csr_matrix


def compute_squic(Y, l):
    '''
    Function to compute SQUIC

    Return:
    - X: adjacency matrix
    - time: computation time
    '''
    [X,_,times,_,_,_] = squic.run(Y,l)

    time = times[0]

    # print(X.todense())
    return X, time


def squic_fit(Y, lambda_val, eta, kappa=0, tau=0):
    '''
    Function to compute SQUIC-Fit

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


def squic_fit_matrix(Y, l, matrix, tau=0):
    '''
    Function to pass a matrix as bias to SQUIC

    Return:
    - X_final: adjacency matrix
    - end_time: computation time
    '''
    start_time = time.time()

    M_sparse = csr_matrix(matrix)

    # Step 4: Second SQUIC estimation with bias (Equation 11)
    X2, _, _, _, _, _ = squic.run(Y, l, M=M_sparse)
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
    
    end_time = round(time.time() - start_time, 2)
    
    return X_final, end_time

def squic_fit_matrix2(Y, l, matrix, tau=0):
    start_time = time.time()

    M_sparse = csr_matrix(matrix)

    # Step 4: Second SQUIC estimation with bias (Equation 11)
    X2, _, _, _, _, _ = squic.run(Y, l, M=M_sparse)
    Theta2 = X2.todense()

    # Step 5: Construct the final M-matrix
    Theta_final = np.copy(Theta2)

    # Mask out values where Theta2 < -tau
    mask = (Theta2 > -tau) & np.triu(np.ones_like(Theta2), k=1).astype(bool)
    Theta_final[mask] = 0

    # Restore diagonal
    np.fill_diagonal(Theta_final, np.diag(Theta2))

    end_time = round(time.time() - start_time, 2)
    
    return Theta_final, end_time
    
def count_nnz(X, rows):
    '''
    Function to count nnz on a matrix
    
    Returns:
    - nnz: number of non-zero elements
    - nnz_r: number of non-zero elements per row
    '''
    nnz = X.nnz
    nnz_r = nnz / rows
    return nnz, nnz_r

def nnz_fit(X, rows):
    '''
    Function to count nnz on a matrix
    
    Returns:
    - nnz: number of non-zero elements
    - nnz_r: number of non-zero elements per row
    '''
    A = sp.sparse.csr_array(X)
    nnz = A.count_nonzero()
    nnz_r = nnz / rows
    return nnz, nnz_r

def sparsity_pattern(X, save=False, path=None):
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
