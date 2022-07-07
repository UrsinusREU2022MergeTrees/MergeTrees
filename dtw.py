import numpy as np
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython=True)
def cdtw(X, Y, radius=-1):
    """
    Dynamic time warping with constraints, as per the Sakoe-Chiba band
    
    Parameters
    ----------
    X: ndarray(M)
        A time series with M points
    Y: ndarray(N)
        A time series with N points
    radius: int
        How far away a warping path is allowed to stray from the identity
        map in either direction
    
    Returns
    -------
        (float: cost, ndarray(K, 2): The warping path)
    """
    M = X.size
    N = Y.size
    if radius == -1:
        radius = M
        if N > radius:
            radius = N
    if radius < 1:
        radius = 1

    ## Step 1: Dynamic programming steps
    S = np.zeros((M, N), dtype=np.float32) # Dyn prog matrix
    P = np.zeros((M, N), dtype=np.int32) # Path pointer matrix
    S[0, 0] = np.abs(X[0]-Y[0])
    for i in range(M):
        for j in range(N):
            if i > 0 or j > 0:
                if abs(i-j) > radius:
                    S[i, j] = np.inf
                else:
                    neighbs = [np.inf, np.inf, np.inf]
                    idx = 0
                    if j > 0: # Left
                        neighbs[0] = S[i, j-1]
                    if i > 0: # Up
                        neighbs[1] = S[i-1, j]
                        if neighbs[1] < neighbs[idx]:
                            idx = 1
                    if i > 0 and j > 0: # Diag
                        neighbs[2] = S[i-1, j-1]
                        if neighbs[2] < neighbs[idx]:
                            idx = 2
                    S[i, j] = neighbs[idx] + np.abs(X[i]-Y[j])
                    P[i, j] = idx
    # Step 2: Backtracing
    i = M-1
    j = N-1
    path = [[i, j]]
    step = [[0, -1], [-1, 0], [-1, -1]] # LEFT, UP, DIAG
    while not(path[-1][0] == 0 and path[-1][1] == 0):
        s = step[P[i, j]]
        i += s[0]
        j += s[1]
        path.append([i, j])
    path.reverse()
    return (S[-1, -1], S, path)