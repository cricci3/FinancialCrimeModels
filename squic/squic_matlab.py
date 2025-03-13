import numpy as np
import scipy.sparse as sp
import squic

def squic_matlab(Y, lambda_val, eta):
    # Get dimensions
    p, _ = Y.shape
    
    # Construct initial Lambda matrix (zero sparse matrix)
    Lambda_matrix = sp.csr_matrix((p, p))

    # W0 = Lambda_matrix
    # X0 = Lambda_matrix
    
    print('Squic Run 1: Estimate graphical bias of negatively correlated variables')
    
    # First SQUIC call
    X1, _, _, _, _, _ = squic.run(Y, lambda_val, M=Lambda_matrix)
    
    # Remove diagonal entries and select structure of negative off-diagonal entries
    G_neg = X1.todense() - np.diag(np.diag(X1.todense()))
    G_neg = (G_neg < 0).astype(float)
    
    print('Squic Run 2: Use the graphical bias and select negatively correlated variables')
    
    # Construct Lambda matrix
    Lambda_matrix = sp.csr_matrix(eta * G_neg)
    
    # Second SQUIC call
    X2,  _, _, _, _, _ = squic.run(Y, lambda_val, M=Lambda_matrix)
    
    # Remove diagonal entries
    W = X2.todense() - np.diag(np.diag(X2.todense()))
    
    # Create the adjacency matrix with weights
    #  Select the negative off-diagonal entries
    W = np.abs(W) * (W < 0)
    
    return W