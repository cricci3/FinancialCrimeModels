import numpy as np
import squic_folder


def squic_fit(Y, l, eta, kappa, tau):
    # # First squic call -> Identify negative off-diagonal elements (Equation 9)
    X1, _, _, _, _, _ = squic.run(Y, l)
    Theta1 = X1.todense()
    
    # Step 2: Build Graphical Bias G (Equation 10)
    G = np.zeros_like(Theta1)
    G[np.triu_indices_from(G, k=1)] = (Theta1[np.triu_indices_from(Theta1, k=1)] < -kappa).astype(int)
    G += G.T  # Make symmetric
    
    # Step 3: Build Regularization Parameter Matrix Λ (Equation 12)
    Lambda = np.full_like(Theta1, l) 
    # Apply eta where G is nonzero
    Lambda[G != 0] = eta
    
    # Step 4: Second SQUIC estimation with bias (Equation 11)
    X2, _, _, _, _, _ = squic.run(Y, l, M=Lambda)
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