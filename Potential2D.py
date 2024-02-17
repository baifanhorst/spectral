import numpy as np

def index_2D_to_1D(i,j,M):
    # Convert the 2D index (i,j) to the corresponding 1D index
    # The original 2D array is u_{ij}: i=0,...,N-1, j=0,...,M-1
    # The 2D array is reorganized into a 1D vector:
    # (u00, u01, ..., u0M-1,   u10, u11, ...,u1M-1, ..., uN-10, uN-11, ..., uN-1M-1)
    # The 1D index n=iM+j
    return i*M + j

def index_1D_to_2D(n,M):
    # Convert the 1D index n to the corresponding 2D index (i,j)
    # The original 2D array is u_{ij}: i=0,...,N-1, j=0,...,M-1
    # The 2D array is reorganized into a 1D vector:
    # (u00, u01, ..., u0M-1,   u10, u11, ...,u1M-1, ..., uN-10, uN-11, ..., uN-1M-1)
    # The 1D index n=iM+j
    i = n // M
    j = n % M
    return i,j

def MatEqnConverter(A, B, S, N, M):
    # Convert the matrix equation of the form A U + U B = S into C x = d
    # U: unknowns, N * M
    # U_{ij}: i=0,...,N-1, j=0,...,M-1
    # S: known source term, N*M
    # S_{ij}: i=0,...,N-1, j=0,...,M-1
    # A: N * N
    # B: M * M
    # The shapes of the input must be correct, there is no check against wrong inputs
    
    NM = N*M
    
    C = np.zeros((NM, NM))
    d = np.zeros(NM)
    
    for i in range(N):
        for j in range(M):
            n = index_2D_to_1D(i,j,M)
            # Find d[n]
            d[n] = S[i,j]
            # Find C[n,:]
            for k in range(N):
                C[n, index_2D_to_1D(k,j,M)] += A[i,k]
            for k in range(M):
                C[n, index_2D_to_1D(i,k,M)] += B[k,j]    
    
    return C, d


def MatEqnConverter2(A, B, S, N, M):
    # Convert the matrix equation of the form A U + U B = S into C x = d
    # C is constructed with the Kronecker product
    # U: unknowns, N * M
    # U_{ij}: i=0,...,N-1, j=0,...,M-1
    # S: known source term, N*M
    # S_{ij}: i=0,...,N-1, j=0,...,M-1
    # A: N * N
    # B: M * M
    # The shapes of the input must be correct, there is no check against wrong inputs
    
    
    C = np.kron(A, np.identity(M)) + np.kron(np.identity(N), B.T)
    d = S.reshape(N*M)
    
    return C, d



def Vector_1D_to_2D(x, N, M):
    # Convert a vector x of length N*M into a matrix of shape (N,M)
    return x.reshape(N, M)


def MatEqnSolver(A, B, S, N, M, Converter=MatEqnConverter):
    C, d = Converter(A, B, S, N, M)
    u = np.linalg.solve(C, d)
    U = Vector_1D_to_2D(u, N, M)
    return U


    
    
    
    
    
    