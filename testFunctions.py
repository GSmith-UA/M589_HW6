import numpy as np
import powerMethod_2D as pm2D
import customAlgorithm as ca

# Some testting functions which are nice to have
def test_complex_2D():
    # This is the companion matrix for (x-2i)(x+2i)(x-1)
    A = np.array([[0,1,0],[0,0,1],[4,-4,1]])
    lam1,lam2,v1,v2,goodExit = pm2D.powerMethod_2D(A,np.array([1,0,3]),np.array([0,1,0]))
    print("EigenValues = ",[lam1,lam2])
    print("EigenVector 1 = ", v1)
    print("EigenVector 2 = ", v2)
    print("Good Exit? ", goodExit)
    return None

def test_custom_Algorithm():
    # This is the companion matrix for (x-2i)(x+2i)(x-1)
    A = np.array([[0,1,0],[0,0,1],[4,-4,1]])
    lam, eigVec, goodExit = ca.customAlgorithm(A)
    print("EigenValues = ",lam)
    print("EigenVector = ", eigVec)
    print("Good Exit? ", goodExit)
    return None