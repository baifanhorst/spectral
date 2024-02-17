import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift

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

def CalcRHS_Burgers_DFT(t, u, nu):
    # Fourier collocation method for solving advection-diffusion equation
    # Evaluating the right-hand side with DFT
    # The RHS for the current problem does not contain t explicitly.
    # The argument t is included just for format consistency with scipy
    ux = diff_DFT(u, 1)
    uxx = diff_DFT(u, 2)
    return nu * uxx - ux * u