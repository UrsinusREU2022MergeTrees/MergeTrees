import numpy as np
from numba import jit
from numba.types import int_, float32
import matplotlib.pyplot as plt
from scipy import sparse

@jit(nopython=True)
def cdtw_(X, Y, radius=-1, compute_path=True):
    """
    Numba helper method for cdtw
    
    Parameters
    ----------
    X: ndarray(M)
        A time series with M points
    Y: ndarray(N)
        A time series with N points
    radius: int
        How far away a warping path is allowed to stray from the identity
        map in either direction
    compute_path: bool
        Whether to compute the warping path
    
    Returns
    -------
        S: List of rows of dynamic programming matrix
        jstarts: j offset of each row
        path: warping path
    """
    M = int(X.size)
    N = int(Y.size)
    if radius == -1:
        radius = M
        if N > radius:
            radius = N
    if radius < 1:
        radius = 1

    ## Step 1: Figure out contrained bounds
    S = [] # Dyn prog matrix
    P = [] # Path backpointer matrix
    jstarts = np.zeros(M, int_)
    jends = np.zeros(M, int_)
    for i in range(M):
        jstart = int(i*N/M - radius)
        if jstart < 0:
            jstart = 0
        jstarts[i] = jstart
        jend = int(i*N/M + radius)
        if jend > N-1:
            jend = N-1
        jends[i] = jend
        S.append(np.zeros(jend-jstart+1, float32))
        if compute_path:
            P.append(np.zeros((jend-jstart+1, 2), int_))

    ## Step 2: Dynamic programming steps
    S[0][0] = np.abs(X[0]-Y[0])
    for i in range(M):
        # Compute all dynamic programming entries in this row
        for jrel in range(len(S[i])):
            if i == 0 and jrel == 0:
                continue
            neighbs = [np.inf, np.inf, np.inf]
            idx = 0
            jup = -1 # Relative index of j in row above this
            if compute_path:
                P[i][jrel] = [0, 0]
            if i > 0:
                jup = jstarts[i] + jrel - jstarts[i-1]
            jup_left = jup - 1
            # Left
            if jrel > 0: 
                neighbs[0] = S[i][jrel-1]
                if compute_path:
                    P[i][jrel] = [i, jrel-1]
            # Up
            if i > 0 and jup >= 0 and jup <= jends[i-1]: 
                neighbs[1] = S[i-1][jup]
                if neighbs[1] < neighbs[idx]:
                    idx = 1
                    if compute_path:
                        P[i][jrel] = [i-1, jup]
            # Diag
            if i > 0 and jup_left >= 0 and jup_left <= jends[-1]: 
                neighbs[2] = S[i-1][jup_left]
                if neighbs[2] < neighbs[idx]:
                    idx = 2
                    if compute_path:
                        P[i][jrel] = [i-1, jup_left]
            S[i][jrel] = neighbs[idx] + np.abs(X[i]-Y[jrel+jstarts[i]])
    # Step 2: Backtracing
    path = None
    if compute_path:
        i = M-1
        j = len(S[i])-1
        path = [[M-1, N-1]]
        while not(path[-1][0] == 0 and path[-1][1] == 0):
            [i, j] = P[i][j]
            path.append([i, j+jstarts[i]])
        path.reverse()
    return S, jstarts, path

def cdtw(X, Y, radius=-1, compute_path=True, return_S=False):
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
    compute_path: bool
        Whether to compute the warping path
    return_S: bool
        Whether to return dynamic programming matrix
    
    Returns
    -------

    """
    S, jstarts, path = cdtw_(X, Y, radius=radius, compute_path=compute_path)
    cost = S[-1][-1]
    if compute_path:
        path = np.array(path, dtype=int)
    ret = (cost, path)
    if return_S:
        M = X.size
        N = Y.size
        I = np.concatenate(tuple([i*np.ones(len(S[i])) for i in range(M)]))
        J = np.concatenate(tuple([jstarts[i] + np.arange(len(S[i])) for i in range(M)]))
        V = np.concatenate(tuple(S))
        S = sparse.coo_matrix((V, (I, J)), shape=(M, N))
        ret = (cost, path, S)
    return ret