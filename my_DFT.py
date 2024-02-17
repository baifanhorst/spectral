import numpy as np

def DFT_direct(f, s):
    # Calculating forward and backward DFT by direct summation
    # f: 1D numpy array, representing either the function values or the Fourier coefficients
    # s=1: forward DFT, s=-1 backward
    
    # Extract the length of the input
    N = len(f)
    
    # Initialize
    F = np.zeros(N, dtype=complex)
    
    for k in range(N): #k=0,1,...,N-1
        for j in range(N):
            # Numpy uses 1j to represent imaginary unit
            F[k] += f[j] * np.exp(-1j * 2 * np.pi/N * j * k * s) 
    return F