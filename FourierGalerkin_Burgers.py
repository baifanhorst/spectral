import numpy as np
from scipy import integrate
from scipy.fft import fft, ifft, fftshift, ifftshift

def CalConvSum_Direct(V, W):
    # Evaluate the convolutional sum of V and W by direct summation
    # Vj, Wj, j=-N/2,...,N/2
    # N must be even
    
    N = len(V) - 1
    N2 = N//2
    
    result = np.zeros(V.shape)
    
    for m in range(-N2, N2+1):
        ind_low = np.max((-N2, m-N2))
        ind_max = np.min((N2, N2+m))
        for p in range(ind_low, ind_max+1):
                # python index = math index + N/2
                result[m+N2] += V[p+N2] * W[m-p+N2]
    
    return result



def CalConvSum_DFT(V, W):
    # Calculating the convolutional sum between V and W with DFT
    
    # N must be even
    N = len(V) - 1
    N2 = N//2
    M = 3 * N2 + 1 # M>3N/2
    # M must be even
    if M%2!=0:
        M += 1
    M2 = M//2    
        
    # Padding (adding zeros to both ends)    
    V_ext = np.concatenate((np.zeros(M2-N2), V, np.zeros(M2-N2-1)))
    W_ext = np.concatenate((np.zeros(M2-N2), W, np.zeros(M2-N2-1)))
    
    # Rearranging
    V_ext = ifftshift(V_ext)
    W_ext = ifftshift(W_ext)
    
    # Inv DFT
    v = ifft(V_ext) * M
    w = ifft(W_ext) * M
    
    # Constructing a
    a = v * w
    
    # DFT to get the convolutional sum
    A = fft(a) / M
    A = fftshift(A)
    
    # Extracting the entries
    return A[M2-N2:M2+N2+1]


def CalcInitCond(f, N):
    # Calculating the initial coefficients for Fourier Galerkin
    # This function is the same as that in FourierGalerkin
    
    # N must be even
    if N%2!=0:
        print("Please use even N")
        return
    
    N2 = N//2
    
    # Initialize the coefficients
    # Correspondence of indices: 
    # Math indices: -N/2, ... ,N/2
    # Python indices: 0, ..., N
    # Relation: python index = math index + N/2
    u0 = np.zeros(N+1, dtype=complex)
    
    for k in range(-N2, N2+1):
        integrand_real = lambda x, k: f(x) * np.cos(k * x)
        integrand_imag = lambda x, k: -f(x) * np.sin(k * x)
        result_real = integrate.quad(integrand_real, 0, 2*np.pi, args=(k,))[0] / (2 * np.pi) 
        result_imag = integrate.quad(integrand_imag, 0, 2*np.pi, args=(k,))[0] / (2 * np.pi)
        u0[k+N2] = result_real + 1j * result_imag
        
    return u0

def CalcRHS_Burgers(t, U, N, nu):
    # Calculating the right-hand side of the ode system 
    # U: truncated Fourier coefficients of the solution u, index: -N/2, ... N/2
    # N must be even
    N2 = N//2
    
    # Calculated the truncated coefficients of u_x
    Q = np.arange(-N2, N2+1)
    W = 1j * Q * U
    
    return -CalConvSum_DFT(U, W)-nu * Q * Q * U

def EvalSolution(x, u, N):
    # Direct evaluation the solution at x
    # u: Fourier coefficients at the current time
    # N must be even
    N2 = N//2
    K = np.arange(-N2, N2+1)
    return np.real(np.sum(u * np.exp(1j * K * x)))
    
    
    
        
    
        
     
        
    
