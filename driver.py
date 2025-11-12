import numpy as np
from scipy.linalg import inv
import helperFunctions as hf
import powerMethod_2D as pm2D
import customAlgorithm as ca
import testFunctions as tf

def main():
    HW6_Questions = True
    testFunctions = False

    if testFunctions:
        print("--------------------------------")
        print("Testing powerMethod_2D()")
        tf.test_complex_2D()
        print("--------------------------------")
        print("Testing customAlgorithm()")
        tf.test_custom_Algorithm()
        print("--------------------------------")

    if HW6_Questions:    
        A = hf.constructCompanionMatrix([6,-5,2,-3,0])
        A_inv = inv(A)
    
        r_0,r_1,x_0,x_1,goodExit = pm2D.powerMethod_2D(A_inv,np.array([1,2,3,0,0]),np.array([0,0,0,1,2]))

        if not(goodExit):
            print("Trust these at your own risk")
        print("--------------------------------")
        print("Power Method 2D for HW 6")
        print("Eigen Value = ", 1/r_0)
        print("Eigen Vector = ", x_0)
        print("Eigen Value = ", 1/r_1)
        print("Eigen Vector = ", x_1)
        print("--------------------------------")
        print("Custom Algorithm for HW 6")
        lam,eigVec,goodExit = ca.customAlgorithm(A_inv)
        print("Eigen Value = ",1/lam)
        print("Eigen Vector = ", eigVec)

    return None
main()