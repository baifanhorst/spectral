import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift

def CalcFourierDiffMat(N):
    # Calculating the Fourier Differentiation Matrix
    # Fourier nodes: xj = 2 * pi / N * j, j=0,1,...,N-1
    
    # Initialization
    D = np.zeros((N,N))
    
    # Calculating the entries
    for i in range(N):
        for j in range(N):
            # Off-diagonal entries
            if i!=j:
                D[i,j] = 0.5 * (-1)**(i-j) / np.tan(np.pi/N*(i-j))
        # Diagonal entries, negative sum trick
        D[i,i] = -np.sum(D[i,:])
    return D

def CalcRHS_AdvDif_DiffMat(t, u, D, nu):
    # Fourier collocation method for solving advection-diffusion equation
    # Evaluating the right-hand side with the differentiation matrix
    # The RHS for the current problem does not contain t explicitly.
    # The argument t is included just for format consistency with scipy
    F = nu * np.dot(D, u)
    return np.dot(D, F-u)


def ODE_SingleStep_RK3_SettingConsts():
    # Setting the auxiliary constants for the Runge-Kutta solver
    ct = np.array([0, 1/3, 3/4]) # currently, these constants are not needed
    cu = np.array([1/3, 15/16, 8/15])
    cg = np.array([0, -5/9, -153/128])
    return ct, cu, cg

def ODE_SingleStep_RK3(u, dt, D, nu, ct, cu, cg):
    # A single step update in the 3rd order Runge-Kutta
    
    # Initialize g
    # Otherwise, there would be error since g is referred in the update of g
    g = np.zeros(u.shape)
    for i in range(3):
        # Use 0 for t in CalcRHS_AdvDif_DiffMat, which is dummy
        g = CalcRHS_AdvDif_DiffMat(0, u, D, nu) + cg[i] * g  
        u = u + cu[i] * dt * g
    return u

def ODE_RK3(u0, dt, tend, nu):
    # Solving the ODE system
    # u0: initial condition
    # dt: time step
    # tend: end time
    # nu: viscosity
    
    # Number of time steps
    Nt = int(np.floor(tend/dt))
    
    # Auxiliary constants
    ct, cu, cg = ODE_SingleStep_RK3_SettingConsts()
    
    # Differentiation matrix
    N = len(u0)
    D = CalcFourierDiffMat(N)
    
    # Iterations
    u = u0
    for it in range(Nt):
        u = ODE_SingleStep_RK3(u, dt, D, nu, ct, cu, cg)
    
    return u


def diff_DFT(u, m):
    # Calculate the derivatives at Fourier nodes by DFT
    # u: function values at Fourier nodes
    # m: order of the derivatives
    
    N = len(u)
    
    if N%2 != 0:
        print('Please use even length')
        return
    
    # DFT
    # Note that here we do not rescale the result by 1/N
    # Correspondingly, when using ifft, we do not rescale, either.
    U = fft(u)
    
    # Rearrangement
    U = fftshift(U)
    
    # Adjustment
    if m%2 != 0:
        U[0] == 0
    
    # Wave numbers, k = -N/2, ..., N/2-1
    K = np.arange(-N//2, N//2)
    
    # Differentiation
    U_deri = (1j*K)**m * U
    U_deri = ifftshift(U_deri) # rearrangement again, very important
    u_deri = ifft(U_deri)
    u_deri = np.real(u_deri)
    
    return u_deri

def CalcRHS_AdvDif_DFT(t, u, nu):
    # Fourier collocation method for solving advection-diffusion equation
    # Evaluating the right-hand side with DFT
    # The RHS for the current problem does not contain t explicitly.
    # The argument t is included just for format consistency with scipy
    ux = diff_DFT(u, 1)
    uxx = diff_DFT(u, 2)
    return nu * uxx - ux
    