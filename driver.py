import numpy as np
from scipy.linalg import inv
import cmath

def main():
    A = constructCompanionMatrix([6,-5,2,-3,0])
    A_inv = inv(A)
    
    r_0,r_1,x_0,x_1,goodExit = powerMethod_2D(A_inv,np.array([1,2,3,0,0]),np.array([0,0,0,1,2]))

    if not(goodExit):
        print("Trust these at your own risk")

    print("Eigen Value = ", 1/r_0)
    print("Eigen Vector = ", x_0)
    print("Eigen Value = ", 1/r_1)
    print("Eigen Vector = ", x_1)
    print("--------------------------------")
    lam,eigVec,goodExit = customAlgorithm(A_inv)
    print("Eigen Value = ",1/lam)
    print("Eigen Vector = ", eigVec)
    print("--------------------------------")
    test_complex_2D()
    print("--------------------------------")
    test_custom_Algorithm()
    print("--------------------------------")
    return None


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

        orthog = gramSchmidt(np.column_stack((y_0,y_1)))
        q_0 = orthog[:,0]
        q_1 = orthog[:,1]

        B = np.array([
            [np.conjugate(q_0).T @ A @ q_0, np.conjugate(q_0).T @ A @ q_1],
            [np.conjugate(q_1).T @ A @ q_0, np.conjugate(q_1).T @ A @ q_1]])
        
        charPolyCoeff = [1,-1*(B[0,0]+B[1,1]),(B[0,0]*B[1,1] - B[1,0]*B[0,1])]
        r_plusNew,r_minusNew = quadSolve(charPolyCoeff)

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


def normalizeVec(x):
    r = np.linalg.norm(x)
    # assert(r > 1e-12, "Zero or near-zero vector in normalizeVec")
    return x / r

def gramSchmidt(A):
    # Takes in a np.array matrix whose columns are the vectors we will orthogonalize...
    rows,cols = np.shape(A)
    A_orthog = np.zeros((rows, cols))
    # rows gives the dimension of the vectors
    # cols gives the number of vectors we will orthogonalize

    # This may NOT be the best Gram-Schmidt implementation, but I want something easy to call...
    q_0 = A[:,0]
    q_0 = normalizeVec(q_0)
    A_orthog[:,0] = q_0

    for i in range(1,cols):
        q = A[:,i]
        # Going to loop through all the ones that are complete....
        for k in range(0,i):
            projectionScalar = np.vdot(q,A_orthog[:,k])
            q = q - projectionScalar*A_orthog[:,k]
        #Last step is to orthog and store in A_orthog...
        q = normalizeVec(q)
        A_orthog[:,i] = q
    return A_orthog

def quadSolve(coeff):
    # Coeffs need to be in order a,b,c or a_2,a_1,a_0
    x_1 = (-coeff[1] + cmath.sqrt(coeff[1]**2 - 4*coeff[0]*coeff[2]))/(2*coeff[0])
    x_2 = (-coeff[1] - cmath.sqrt(coeff[1]**2 - 4*coeff[0]*coeff[2]))/(2*coeff[0])
    return x_1,x_2

def constructCompanionMatrix(coeffVec):
    # Returns an np matrix that is the companion matrix for the given polynomial coeff
    # Coefficients must be ordered as [c_0,c_1,c_2,c_3,...c_{n-1}]: this is the users responsibility
    # Note that these polynomials must be monic so the c_n coeff need not be given
    n = len(coeffVec)
    A = np.eye(n - 1, dtype=float) # Start with an identity matrix
    # Now I concatenate the first col and last row
    A = np.concatenate((np.zeros((n-1,1),dtype=float),A),axis=1)
    A = np.concatenate((A,np.array(coeffVec,ndmin=2)),axis=0)
    # Negate the last row...
    A[n-1,:] = -1*A[n-1,:]
    return A

def powerMethod_standard(A, xInit, maxIter = 10000, tol=1e-9):
    # Runs singular instance of power method to return the leading eigenvalue
    # Also returns boolean to check for good vs bad exit
    eig = None
    goodExit = False
    x_current = np.array(xInit,dtype=complex,ndmin=2)
    
    for k in range(0,maxIter):
        y = A@x_current
        x_new = (1/np.linalg.norm(y))*y
        eigNew = np.vdot(x_new, A@x_new)
        # eigNew = np.asscalar(eigNew)

        # This checks to see how much our eigenvalue is changing... if only a little we have probably converged
        # Consider what the tolerance should be here... maybe better to check the eigenvector??
        if (k>0) and (np.abs(eigNew - eig) < tol):
            eig = eigNew
            goodExit = True
            return eig, x_new, goodExit
        
        eig = eigNew
        x_current = x_new

    return eig, np.ndarray.flatten(x_new), goodExit

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


def test_complex_2D():
    # This is the companion matrix for (x-2i)(x+2i)(x-1)
    A = np.array([[0,1,0],[0,0,1],[4,-4,1]])
    lam1,lam2,v1,v2,goodExit = powerMethod_2D(A,np.array([1,0,3]),np.array([0,1,0]))
    print("EigenValues = ",[lam1,lam2])
    print("EigenVector 1 = ", v1)
    print("EigenVector 2 = ", v2)
    print("Good Exit? ", goodExit)
    return None

def test_custom_Algorithm():
    # This is the companion matrix for (x-2i)(x+2i)(x-1)
    A = np.array([[0,1,0],[0,0,1],[4,-4,1]])
    lam, eigVec, goodExit = customAlgorithm(A)
    print("EigenValues = ",lam)
    print("EigenVector = ", eigVec)
    print("Good Exit? ", goodExit)
    return None
main()