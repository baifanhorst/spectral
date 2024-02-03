import numpy as np
from scipy import integrate
from scipy.fft import fft, ifft, fftshift, ifftshift

def CalcInitCond(f, N):
    # Calculating the initial coefficients for Fourier Galerkin
    
    # N must be even
    if N%2!=0:
        print("Please use even N")
        return
    
    N2 = N//2
    
    # Initialize the coefficients
    # Correspondence of indices: 
    # Original indices: -N/2, ... ,N/2
    # Current indices: 0, ..., N
    # Relation: ind_new = ind_old + N/2
    u0 = np.zeros(N+1, dtype=complex)
    
    for k in range(-N2, N2+1):
        integrand = lambda x, k: f(x) * np.exp(-1j * k * x) 
        u0[k+N2] = integrate.quad(integrand, 0, 2*np.pi, args=(k,))[0] / (2 * np.pi) 
        
    return u0


def CalcRHS_AdvDif(t, u, N, nu):
    # Calculating the right-hand side of the ode system 
    # N must be even
    N2 = N//2
    K = np.arange(-N2, N2+1)
    return -(1j * K + nu * K * K) * u


def EvalSolution(x, u, N):
    # Direct evaluation the solution at x
    # u: Fourier coefficients at the current time
    # N must be even
    N2 = N//2
    K = np.arange(-N2, N2+1)
    return np.real(np.sum(u * np.exp(1j * K * x)))

def EvalSolution_DFT(coeff, N):
    # Evaluation of the solution by DFT
    # This can only get the values of the solution at the Fourier nodes:
    # xj: j=0,1,...,N-1
    # N must be even
    # coeff: Fourier coefficients at the current time
    N2 = N//2
    
    U = np.concatenate((coeff[N2:-1], coeff[0:N2]))
    U[N2] *= 2 # This corresponds to wavenumber=-N/2
    
    # inv DFT
    return np.real(ifft(U) * N)
   

def CalcInitCond_DFT(f, N):
    # Using the Fourier coefficients of the Fourier interpolating polynomial as IC
    # instead of Fourier truncated series
    # These coefficients can be obtained by DFT
    
    # Getting the Fourier nodes
    J = np.arange(0,N)
    x = 2 * np.pi / N * J
    f_node_values = f(x)
    
    # DFT
    coeff = fft(f_node_values) / N
    # Rearrange so that the wave numbers are: -N/2, ..., N/2-1
    coeff = fftshift(coeff)
    # Add the coefficient with wave number N/2, which is equal to that of -N/2
    coeff = np.concatenate((coeff, coeff[:1]))
    # The first and the last coefficients must be reduced by half
    # according to the interpolation formular
    coeff[0] /= 2
    coeff[-1] /= 2
    
    return coeff


        