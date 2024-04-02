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
# Functions for the nodal Galerkin method in a square
##########################################################

def MatEqnConverter_Square_NodalGalerkin_Dirichlet(Gx, Gy, wx, wy, RHS, N, M):
    # Constructing the matrix and rhs for the 2D potential problem in a square
    # with Dirichlet boundary conditions, using the nodal Galerkin method
    
    # The unknowns are inner values of U, which is (N+1) * (M+1)
    # The inner values are: U_{ij}: i=1,...,N-1, j=1,...,M-1
    
    # Gx Gy: the matrices appear in the formula
    
    # wx, wy: Legendre-Gauss-Lobatto nodes
    
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
            for n in range(1,N): # n=1,...,N-1
                ind_2nd = index_2D_to_1D_ver2(n,j,M)
                C[ind_1st, ind_2nd] += wy[j] * Gx[i,n]
                
            for m in range(1,M): # m=1,...,M-1
                ind_2nd = index_2D_to_1D_ver2(i,m,M)
                C[ind_1st, ind_2nd] += wx[i] * Gy[j,m]
    
    return C, d

def MatEqnSolver_Square_NodalGalerkin(C, d, N, M):
    # Solving the matrix Cu = d and reshape the solution
    u = np.linalg.solve(C, d)
    return u.reshape(N-1, M-1)



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



#############################################################
# Functions for the nodal Galerkin method in nonsquare domain
#############################################################
def cal_Coefficients_A_Nonsquare_NodalGalerkin_Dirichlet(i, j, k, pars):
    # The nodal Galerkin method
    # Computing the coefficients A_k^{(i,j)} of the algebraic system
    # for the 2D potential problem in nonsquare domain
    # with Dirichlet boundary conditions
    
    # pars = (w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M)
    
    # w_xi: Legendre-Gauss-Lobatto points in the xi direction
    # w_eta: Legendre-Gauss-Lobatto points in the eta direction
    
    # D_xi: (N+1)*(N+1) differentiation matrix wrt xi
    # D_eta: (M+1)*(M+1) differentiation matrix wrt eta
    
    
    # coeff_xi = (X_xi^2 + Y_xi^2)/J
    # coeff_eta = (X_eta^2 + Y_eta^2)/J
    # coeff_mixed = (X_xi X_eta + Y_xi Y_eta)/J
    
    # N, M: numbers of nodes in the xi and eta directions.
    
    w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M = pars
    
    A = 0
    for n in range(0,N+1): # n=0,...,N
        A += - w_xi[n] * w_eta[j] * coeff_eta[n,j] * D_xi[n,k] * D_xi[n,i]
    
    return A

def cal_Coefficients_B_Nonsquare_NodalGalerkin_Dirichlet(i, j, n, s, pars):
    # The nodal Galerkin method
    # Computing the coefficients B of the algebraic system
    # for the 2D potential problem in nonsquare domain
    # with Dirichlet boundary conditions
    
    # pars = (w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M)
    
    # w_xi: Legendre-Gauss-Lobatto points in the xi direction
    # w_eta: Legendre-Gauss-Lobatto points in the eta direction
    
    # D_xi: (N+1)*(N+1) differentiation matrix wrt xi
    # D_eta: (M+1)*(M+1) differentiation matrix wrt eta
    
    
    # coeff_xi = (X_xi^2 + Y_xi^2)/J
    # coeff_eta = (X_eta^2 + Y_eta^2)/J
    # coeff_mixed = (X_xi X_eta + Y_xi Y_eta)/J
    
    # N, M: numbers of nodes in the xi and eta directions.
    
    w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M = pars
    
    B = w_xi[n] * w_eta[j] * coeff_mixed[n,j] * D_eta[j,s] * D_xi[n,i]
    
    return B

def cal_Coefficients_C_Nonsquare_NodalGalerkin_Dirichlet(i, j, k, m, pars):
    # The nodal Galerkin method
    # Computing the coefficients C_{km}^{(i,j)} of the algebraic system
    # for the 2D potential problem in nonsquare domain
    # with Dirichlet boundary conditions
    
    # pars = (w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M)
    
    # w_xi: Legendre-Gauss-Lobatto points in the xi direction
    # w_eta: Legendre-Gauss-Lobatto points in the eta direction
    
    # D_xi: (N+1)*(N+1) differentiation matrix wrt xi
    # D_eta: (M+1)*(M+1) differentiation matrix wrt eta
    
    
    # coeff_xi = (X_xi^2 + Y_xi^2)/J
    # coeff_eta = (X_eta^2 + Y_eta^2)/J
    # coeff_mixed = (X_xi X_eta + Y_xi Y_eta)/J
    
    # N, M: numbers of nodes in the xi and eta directions.
    
    w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M = pars
    
    C = w_xi[i] * w_eta[m] * coeff_mixed[i,m] * D_xi[i,k] * D_eta[m,j] 
    
    return C


def cal_Coefficients_D_Nonsquare_NodalGalerkin_Dirichlet(i, j, s, pars):
    # The nodal Galerkin method
    # Computing the coefficients D_s^{(i,j)} of the algebraic system
    # for the 2D potential problem in nonsquare domain
    # with Dirichlet boundary conditions
    
    # pars = (w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M)
    
    # w_xi: Legendre-Gauss-Lobatto points in the xi direction
    # w_eta: Legendre-Gauss-Lobatto points in the eta direction
    
    # D_xi: (N+1)*(N+1) differentiation matrix wrt xi
    # D_eta: (M+1)*(M+1) differentiation matrix wrt eta
    
    
    # coeff_xi = (X_xi^2 + Y_xi^2)/J
    # coeff_eta = (X_eta^2 + Y_eta^2)/J
    # coeff_mixed = (X_xi X_eta + Y_xi Y_eta)/J
    
    # N, M: numbers of nodes in the xi and eta directions.
    
    w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M = pars
    
    D = 0
    for m in range(0,M+1): # m=0,...,M+1
        D += - w_xi[i] * w_eta[m] * coeff_xi[i,m] * D_eta[m,s] * D_eta[m,j]
    
    return D


def cal_RHS_Nonsquare_NodalGalerkin_Dirichlet(U, S, J, w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M):
    # Computing the right-hand side obtained by moving all known terms
    # U: the solution matrix, only the boundary values are used
    # S: Source term
    # J: Jacobi
    
    # w_xi: Legendre-Gauss-Lobatto points in the xi direction
    # w_eta: Legendre-Gauss-Lobatto points in the eta direction
    
    # D_xi: (N+1)*(N+1) differentiation matrix wrt xi
    # D_eta: (M+1)*(M+1) differentiation matrix wrt eta
    
    
    # coeff_xi = (X_xi^2 + Y_xi^2)/J
    # coeff_eta = (X_eta^2 + Y_eta^2)/J
    # coeff_mixed = (X_xi X_eta + Y_xi Y_eta)/J
    
    # N,M : numbers of nodes in the xi and eta directions
    
    
    RHS = J * S
    
    pars = (w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M)
    
    for i in range(1, N): # i = 1,...,N-1
        for j in range(1, M): # j = 1,...,M-1
            RHS[i,j] *= w_xi[i] * w_eta[j]
            
            for k in (0, N):
                # Computing A_k^{(i,j)}
                A = cal_Coefficients_A_Nonsquare_NodalGalerkin_Dirichlet(i, j, k, pars)
                RHS[i,j] -= A * U[k,j]
            
            for n in (0, N):
                for s in range(0, M+1): # s = 0,...,M
                    # Computing B_{ns}^{(i,j)}
                    B = cal_Coefficients_B_Nonsquare_NodalGalerkin_Dirichlet(i, j, n, s, pars)
                    RHS[i,j] -= B * U[n,s]
                    
            for n in range(1,N): # n=1,...,N-1
                for s in (0,M):
                    # Computing B_{ns}^{(i,j)}
                    B = cal_Coefficients_B_Nonsquare_NodalGalerkin_Dirichlet(i, j, n, s, pars)
                    RHS[i,j] -= B * U[n,s]
                    
            for k in (0, N):
                for m in range(0, M+1):
                    # Computing C_{km}^{(i,j)}
                    C_ = cal_Coefficients_C_Nonsquare_NodalGalerkin_Dirichlet(i, j, k, m, pars)
                    RHS[i,j] -= C_ * U[k,m]
                    
            for k in range(1, N): # k=1,...,N-1
                for m in (0, M):
                    # Computing C_{km}^{(i,j)}
                    C_ = cal_Coefficients_C_Nonsquare_NodalGalerkin_Dirichlet(i, j, k, m, pars)
                    RHS[i,j] -= C_ * U[k,m]
                    
                    
            for s in (0, M):
                # Computing D_s^{(i,j)}
                D = cal_Coefficients_D_Nonsquare_NodalGalerkin_Dirichlet(i, j, s, pars)
                RHS[i,j] -= D * U[i,s]
                    
    
    return RHS


def MatEqnConverter_Nonsquare_NodalGalerkin_Dirichlet(w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M, RHS):
    # The nodal Galerkin method
    # Converting the algebraic system into the form Cx = d
    # for the 2D potential problem in nonsquare domain
    # with Dirichlet boundary conditions
    
    # w_xi: Legendre-Gauss-Lobatto points in the xi direction
    # w_eta: Legendre-Gauss-Lobatto points in the eta direction
    
    # D_xi: (N+1)*(N+1) differentiation matrix wrt xi
    # D_eta: (M+1)*(M+1) differentiation matrix wrt eta
    
    
    # coeff_xi = (X_xi^2 + Y_xi^2)/J
    # coeff_eta = (X_eta^2 + Y_eta^2)/J
    # coeff_mixed = (X_xi X_eta + Y_xi Y_eta)/J
    
    # RHS: the right-hand side after moving all known terms
    # Note that only inner values of RHS are used: i=1,...,N-1, j=1,...,M-1
    # and the boundary values of RHS are not set and not used.
    # However, the whole RHS must be used as input
    # The shapes of the input must be correct, there is no check against wrong inputs
    
    # N, M: numbers of nodes in the xi and eta directions.
    
    
    num_unknowns = (N-1) * (M-1)
    
    C = np.zeros((num_unknowns, num_unknowns))
    d = np.zeros(num_unknowns)
    
    pars = (w_xi, w_eta, D_xi, D_eta, coeff_xi, coeff_eta, coeff_mixed, N, M)
    
    
    for i in range(1, N): # i=1,...,N-1
        for j in range(1, M): # j=1,...,M-1
            # Compute the row index
            ind_1st = index_2D_to_1D_ver2(i,j,M)
            # Find d[ind_1st]
            d[ind_1st] = RHS[i,j]
            
            for k in range(1, N): # k=1,...,N-1
                # Computing A_k^{(i,j)}
                A = cal_Coefficients_A_Nonsquare_NodalGalerkin_Dirichlet(i, j, k, pars)
                # Compute the column index
                ind_2nd = index_2D_to_1D_ver2(k,j,M)
                # Update C
                C[ind_1st, ind_2nd] += A
            
            for n in range(1, N): # n=1,...,N-1
                for s in range(1, M): # s=1,...,M-1
                    # Computing B_{ns}^{(i,j)}
                    B = cal_Coefficients_B_Nonsquare_NodalGalerkin_Dirichlet(i, j, n, s, pars)
                    # Compute the column index
                    ind_2nd = index_2D_to_1D_ver2(n,s,M)
                    # Update C
                    C[ind_1st, ind_2nd] += B
                    
            for k in range(1, N): # k=1,...,N-1
                for m in range(1, M): # m=1,...,M-1
                    # Computing C_{km}^{(i,j)}
                    # Due to the repetition of variable names, here we use C_
                    C_ = cal_Coefficients_C_Nonsquare_NodalGalerkin_Dirichlet(i, j, k, m, pars)
                    # Computing the column index
                    # Compute the column index
                    ind_2nd = index_2D_to_1D_ver2(k,m,M)
                    # Update C
                    C[ind_1st, ind_2nd] += C_
            
            for s in range(1, M): # s = 1,...,M-1
                # Computing D_s^{(i,j)}
                D = cal_Coefficients_D_Nonsquare_NodalGalerkin_Dirichlet(i, j, s, pars)
                # Compute the column index
                ind_2nd = index_2D_to_1D_ver2(i,s,M)
                # Update C
                C[ind_1st, ind_2nd] += D
                
    return C, d


##################################################################
def MatEqnSolver_Cxd(C, d, N, M):
    # Solving the matrix Cu = d and reshape the solution
    u = np.linalg.solve(C, d)
    return u.reshape(N-1, M-1)
                
                
             
    
    