import numpy as np
import helperFunctions as hf

def powerMethod_2D(A,x_0,x_1,epsilonTol = 1e-9,maxIterations = 10000):
    # Let's make sure that all the dimensions are valid... if something is wrong let's just exit
    assert(np.shape(x_0)[0] == np.shape(x_1)[0])
    assert(np.shape(A)[1] == np.shape(x_0)[0])

    # Note: fixme use caution as we might have problems where eigenvalues are closely clustered...
    r_plusOld=r_minusOld = 1e10
    goodExit = False
    for i in range(0,maxIterations):
        y_0 = A@x_0
        y_1 = A@x_1

        orthog = hf.gramSchmidt(np.column_stack((y_0,y_1)))
        q_0 = orthog[:,0]
        q_1 = orthog[:,1]

        B = np.array([
            [np.conjugate(q_0).T @ A @ q_0, np.conjugate(q_0).T @ A @ q_1],
            [np.conjugate(q_1).T @ A @ q_0, np.conjugate(q_1).T @ A @ q_1]])
        
        charPolyCoeff = [1,-1*(B[0,0]+B[1,1]),(B[0,0]*B[1,1] - B[1,0]*B[0,1])]
        r_plusNew,r_minusNew = hf.quadSolve(charPolyCoeff)

        if (i>0) and abs((r_plusNew - r_plusOld)/r_plusNew)<epsilonTol and abs((r_minusNew - r_minusOld)/r_minusOld)<epsilonTol:
            goodExit = True
            break

        r_plusOld = r_plusNew
        r_minusOld = r_minusNew
        x_0 = q_0
        x_1 = q_1
        
    if not(goodExit):
        print("Max Iterations reached without hitting tolerance: use at users risk")

    return r_plusNew,r_minusNew,x_0,x_1,goodExit