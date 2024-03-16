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


def MatEqnConverter_NodalGalerkin(A, B, S, E, F, N, M):
    # Convert the matrix equation of the form (AU)*E + (UB)*F = S into C x = d
    # U: unknowns, N * M
    # U_{ij}: i=0,...,N-1, j=0,...,M-1
    # S: known source term, N*M
    # S_{ij}: i=0,...,N-1, j=0,...,M-1
    # A: N * N
    # B: M * M
    # E, F: N * M
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
                C[n, index_2D_to_1D(k,j,M)] += A[i,k] * E[i,j]
            for k in range(M):
                C[n, index_2D_to_1D(i,k,M)] += B[k,j] * F[i,j]  
    
    return C, d


def Vector_1D_to_2D(x, N, M):
    # Convert a vector x of length N*M into a matrix of shape (N,M)
    return x.reshape(N, M)


def MatEqnSolver(A, B, S, N, M, Converter=MatEqnConverter):
    C, d = Converter(A, B, S, N, M)
    u = np.linalg.solve(C, d)
    U = Vector_1D_to_2D(u, N, M)
    return U

def MatEqnSolver_NodalGalerkin(A, B, S, E, F, N, M):
    C, d = MatEqnConverter_NodalGalerkin(A, B, S, E, F, N, M)
    u = np.linalg.solve(C, d)
    U = Vector_1D_to_2D(u, N, M)
    return U



##########################################################
# Functions for nonsquare geometry
##########################################################

def index_2D_to_1D_ver2(i,j,M):
    # Convert the 2D index (i,j) to the corresponding 1D index
    # The original 2D array is u_{ij}: i=1,...,N-1, j=1,...,M-1
    # The 2D array is reorganized into a 1D vector:
    # (u11, u12, ..., u1M-1,   u21, u22, ...,u2M-1, ..., uN-1,1, uN-1,2, ..., uN-1,M-1)
    # The 1D index n=(i-1)*(M-1) + (j-1)
    return (i-1)*(M-1) + j-1

def index_1D_to_2D_ver2(n,M):
    # Convert the 1D index n to the corresponding 2D index (i,j)
    # The original 2D array is u_{ij}: i=1,...,N-1, j=1,...,M-1
    # The 2D array is reorganized into a 1D vector:
    # (u11, u12, ..., u1M-1,   u21, u22, ...,u2M-1, ..., uN-1,1, uN-1,2, ..., uN-1,M-1)
    # The 1D index n=(i-1)*(M-1) + (j-1)
    i = n // (M-1) + 1
    j = n % (M-1) + 1
    return i,j


def cal_RHS(D_xi, D_eta, J, coeff_xi, coeff_eta, coeff_mixed, U, S, N, M):
    # Computing the right-hand side for inner nodes
    # Only rhs for inner nodes needs to be calculated
    
    # The original rhs
    RHS = J * S
    
    # Subtracting known terms from lhs
    for i in range(1,N): # i=1,...,N-1
        for j in range(1,M): # j=1,...,M-1
            
            for k in range(0,N+1): # k=0,...,N
                for n in [0,N]: # n=0,N
                    RHS[i,j] -= D_xi[i,k] * coeff_eta[k,j] * D_xi[k,n] * U[n,j]
            
            for k in [0,N]: # k=0,N
                for m in range(0,M+1): # m=0,...,M
                    RHS[i,j] += D_xi[i,k] * coeff_mixed[k,j] * D_eta[j,m] * U[k,m]
                    
            for k in range(1,N): # k=1,...,N-1
                for m in [0,M]: # m=0,M
                    RHS[i,j] += D_xi[i,k] * coeff_mixed[k,j] * D_eta[j,m] * U[k,m]
                    
                    
            for n in [0,N]: # n=0,N
                for l in range(0,M+1): # l=0,...,M
                    RHS[i,j] += D_eta[j,l] * coeff_mixed[i,l] * D_xi[i,n] * U[n,l]
            
            for n in range(1,N): # n=1,...,N-1
                for l in [0,M]: # m=0,M
                    RHS[i,j] += D_eta[j,l] * coeff_mixed[i,l] * D_xi[i,n] * U[n,l]
                    
            for l in range(0,M+1): # l=0,...,M
                for m in [0,M]: # m=0,M
                    RHS[i,j] -= D_eta[j,l] * coeff_xi[i,l] * D_eta[l,m] * U[i,m]
                    
    return RHS
          


def MatEqnConverter_NonsquareCollocation_Dirichlet(D_xi, D_eta, J, coeff_xi, coeff_eta, coeff_mixed, RHS, N, M):
    # Constructing the matrix and rhs for the 2D potential problem in nonsquare domain
    # with Dirichlet boundary conditions
    
    # The unknows are inner values of U, which is (N+1) * (M+1)
    # The unner values are: U_{ij}: i=1,...,N-1, j=1,...,M-1
    
    # D_xi: (N+1)*(N+1) differentiation matrix wrt xi
    # D_eta: (M+1)*(M+1) differentiation matrix wrt eta
    
    # J: nodal Jabobian values, (N+1) * (M+1)
    
    # coeff_xi = (X_xi^2 + Y_xi^2)/J
    # coeff_eta = (X_eta^2 + Y_eta^2)/J
    # coeff_mixed = (X_xi X_eta + Y_xi Y_eta)/J
    
    # RHS: known right-hand side, (N+1)*(M+1)
    # RHS_{ij}: i=0,...,N, j=0,...,M
    # Note that only inner values of RHS are used: i=1,...,N-1, j=1,...,M-1
    # and the boundary values of RHS are not set and not used.
    # However, the whole RHS must be used as input
    # The shapes of the input must be correct, there is no check against wrong inputs
    
    num_unknowns = (N-1) * (M-1)
    
    C = np.zeros((num_unknowns, num_unknowns))
    d = np.zeros(num_unknowns)
    
    for i in range(1,N): # i=1,...,N-1
        for j in range(1,M): # j=1,...,M-1
            ind_1st = index_2D_to_1D_ver2(i,j,M)
            # Find d[ind_1st]
            d[ind_1st] = RHS[i,j]
            # Find C[ind_1st,:]
            for k in range(0,N+1): # k=0,...,N
                for n in range(1, N): # n=1,...,N-1
                    ind_2nd = index_2D_to_1D_ver2(n,j,M)
                    C[ind_1st, ind_2nd] += D_xi[i,k] * coeff_eta[k,j] * D_xi[k,n]
            
            for k in range(1,N): # k=1,...,N-1
                for m in range(1,M): # m=1,...,M-1
                    ind_2nd = index_2D_to_1D_ver2(k,m,M)
                    C[ind_1st, ind_2nd] -= D_xi[i,k] * coeff_mixed[k,j] * D_eta[j,m]
            
            for l in range(1,M): # l=1,...,M-1
                for n in range(1,N): # n=1,...,N-1
                    ind_2nd = index_2D_to_1D_ver2(n,l,M)
                    C[ind_1st, ind_2nd] -= D_eta[j,l] * coeff_mixed[i,l] * D_xi[i,n]
            
            for l in range(0,M+1): #l=0,...,M
                for m in range(1, M): # m=1,...,M-1
                    ind_2nd = index_2D_to_1D_ver2(i,m,M)
                    C[ind_1st, ind_2nd] += D_eta[j,l] * coeff_xi[i,l] * D_eta[l,m]
    
    return C, d


def MatEqnSolver_NonsquareCollocation(C, d, N, M):
    # Solving the matrix Cu = d and reshape the solution
    u = np.linalg.solve(C, d)
    return u.reshape(N-1, M-1)
    
    
    
    