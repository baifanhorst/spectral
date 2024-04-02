import numpy as np

def Jacobi(A, b, maxiter, tol):
    # Solving Ax=b by the Jacobi iteration
    # A: (N,N) 2D numpy array
    # b: (N,) 1D numpy array
    # maxiter: the number of iterations
    # tol: absolute tolerance
    
    N = b.shape[0]
    
    # Initial guess
    x = np.zeros(N)
    # Diagonal entries of A
    diag_A = np.diag(A)
    
    for i in range(maxiter):
        x_new = (b - np.dot(A,x) + diag_A * x) / diag_A
        error_max = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if error_max < tol:
            break
        
    return x, error_max


def GaussSeidel(A, b, maxiter, tol):
    # Solving Ax=b by the Gauss Seidel iteration
    # A: (N,N) 2D numpy array
    # b: (N,) 1D numpy array
    # maxiter: the number of iterations
    # tol: absolute tolerance
    
    N = b.shape[0]
    
    # Initial guess
    x = np.zeros(N)
    
    
    
    for n in range(maxiter):
        # Initialize error
        # The error here is the maximum element difference, i.e. inf norm
        err = 0
        
        for i in range(N): 
            x_old = x[i]
            x[i] = ( b[i] - np.dot(A[i,:], x) + A[i,i] * x[i] ) / A[i,i]
            err_new = np.abs(x[i]-x_old)
            if err_new > err:
                err = err_new
        
        if err < tol:
            break
        
    return x, err


def SOR(A, b, w, maxiter, tol):
    # Solving Ax=b by the Gauss Seidel iteration
    # A: (N,N) 2D numpy array
    # b: (N,) 1D numpy array
    # w: weight, must be in (0,2)
    # maxiter: the number of iterations
    # tol: absolute tolerance
    
    N = b.shape[0]
    
    # Initial guess
    x = np.zeros(N)
    
    for n in range(maxiter):
        # Initialize error
        # The error here is the maximum element difference, i.e. inf norm
        err = 0
        
        for i in range(N): 
            x_old = x[i]
            x[i] = w * ( b[i] - np.dot(A[i,:], x) + A[i,i] * x[i] ) / A[i,i] + (1-w) * x_old
            err_new = np.abs(x[i]-x_old)
            if err_new > err:
                err = err_new
        
        if err < tol:
            break
        
    return x, err


def CG(A, b, maxiter, tol):
    # Conjugated gradient method
    
        
    # Initial guess
    x = np.zeros(b.shape)
    # Initialize direction
    d = b - np.dot(A,x)
    # Initialize residual
    r = b - np.dot(A,x)
    rr = np.dot(r,r)
    
    
    for n in range(maxiter):
        
        # Update x
        Ad = np.dot(A,d)
        a = rr / np.dot(d, Ad)
        x += a * d
        
        
        # Update d
        r -= a * Ad
        rr_new = np.dot(r,r)
        b = rr_new / rr
        d = r + b * d
        
        # Save rr for the next iteration
        rr = rr_new
        
        # Error
        err = np.linalg.norm(r, ord=np.inf)
        if err<tol:
            break
        
    return x, err


def CG_JacobiPrecond(A, b, maxiter, tol):
    # Preconditioned conjugated gradient method
    # The preconditioner is the diagonal of A
    
    # Extract the diagonal of A
    diag_A = np.diag(A)
        
    # Initial guess
    x = np.zeros(b.shape)
    # Initialize residual
    r = b - np.dot(A,x)
    # Initialize direction
    d = r / diag_A
    # Initialize residual of the preconditioned system
    z = r / diag_A
    
    zr = np.dot(z,r)
    
    
    
    for n in range(maxiter):
        
        # Update x
        Ad = np.dot(A,d)
        a = zr / np.dot(d, Ad)
        x += a * d
        
        
        
        # Update d
        r -= a * Ad
        z = r / diag_A
        
        zr_new = np.dot(z,r)
        b = zr_new / zr
        d = z + b * d
        
        # Save rr for the next iteration
        zr = zr_new
        
        # Error
        err = np.linalg.norm(r, ord=np.inf)
        if err<tol:
            break
        
    return x, err

#############################################################
def BICGSTAB(A, b, maxiter):
    # Biconjugate gradient stabilized method
    
    # Initial guess
    x = np.zeros(b.shape)
    
    # Initial residue
    r = b - np.dot(A, x)
    
    # Initial random r'_0
    r_ = b - np.dot(A, x)
    
    # Initial rho
    rho = np.dot(r_, r)
    
    # Initial search direction
    d = b - np.dot(A, x)
    
    for n in range(maxiter):
        # First update of the solution
        v = np.dot(A, d)
        a = rho / np.dot(r_, v)
        h = x + a * d
        s = r - a * v
        # Second update of the solution
        t = np.dot(A, s)
        w = np.dot(t, s) / np.dot(t,t)
        x = h + w * s
        r = s - w * t
        # Update of the search direction
        rho_new = np.dot(r_, r)
        b = rho_new / rho * a / w
        d = r + b * (d - w * v)
        
        rho = rho_new
        
    return x
    
        
        
        
        
        