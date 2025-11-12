import numpy as np

def customAlgorithm(A):
    # Assume that the A here is actually the inverse calculated previously in main()
    gamma_list = [0.1,1,10,100] # One of these should work...
    rows,cols = np.shape(A)
    assert(rows == cols)
    for gamma in gamma_list:
        B = A + gamma*np.eye(rows)*(1j)
        eig, eigVec, goodExit = powerMethod_standard(B,np.ones((rows,1)))
        lam = eig - gamma*(1j)
        if np.abs(np.imag(lam)) > 1e-6:
            print("Gamma = " ,gamma)
            if not(goodExit):
                print("Bad Exit out of power method... use caution!")
            break
    return lam, eigVec, goodExit   