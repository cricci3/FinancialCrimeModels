import matplotlib.pyplot as plt
import squic
import scipy as sp
import numpy as np


def compute_squic(Y, l):
    [X,_,times,_,_,_] = squic.run(Y,l)

    time = times[0]

    print(X.todense())
    return X, time

def squic_fit(Y, lambda_val, eta, kappa=0, tau=0):
    # # First squic call -> Identify negative off-diagonal elements (Equation 9)
    X1, _, _, _, _, _ = squic.run(Y, lambda_val)
    Theta1 = X1.todense()
    
    # Step 2: Build Graphical Bias G (Equation 10)
    G = np.zeros_like(Theta1)
    G[np.triu_indices_from(G, k=1)] = (Theta1[np.triu_indices_from(Theta1, k=1)] < -kappa).astype(int)
    G += G.T  # Make symmetric
    
    # Step 3: Build Regularization Parameter Matrix Λ (Equation 12)
    #Lambda = np.full_like(Theta1, lambda_val) 
    Lambda = np.zeros_like(Theta1)
    # Apply eta where G is nonzero
    Lambda[G != 0] = eta
    
    # Step 4: Second SQUIC estimation with bias (Equation 11)
    X2, _, _, _, _, _ = squic.run(Y, lambda_val, M=Lambda)
    Theta2 = X2.todense()
    
    # Step 5: Construct the final M-matrix (Equation 13)
    Theta_final = np.zeros_like(Theta2)
    
    # Get diagonal
    diag = np.diag(Theta2)
    
    # Identify negative off-diagonal elements
    n = Theta2.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if Theta2[i, j] < -tau:
                Theta_final[i, j] = Theta2[i, j]
                Theta_final[j, i] = Theta2[j, i]
    
    # Restore diagonal
    np.fill_diagonal(Theta_final, diag)
    
    return Theta_final
    
def count_nnz(X, rows):
    # Count nonzero elements in X
    nnz = X.nnz
    nnz_r = nnz / rows
    return nnz, nnz_r

def sparsity_pattern(X):
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
    plt.show()

def nnz_fit(theta, rows):
    A = sp.sparse.csr_array(theta)
    nnz = A.count_nonzero()
    nnz_r = nnz / rows
    return nnz, nnz_r