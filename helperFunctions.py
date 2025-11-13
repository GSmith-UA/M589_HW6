import numpy as np
import cmath
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