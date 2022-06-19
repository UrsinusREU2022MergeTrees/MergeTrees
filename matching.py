import numpy as np
#from numba import jit
from mergetree import *

def weight_sequence_distance(w1, w2):
    """
    Compute the edit distance between non-redundant weight sequences

    Parameters
    ----------
    w1: ndarray(N)
        First weight sequence
    w2: ndarray(N)
        Second weight sequence
    
    Returns
    -------
    float: Optimal distance
    list of list: Matched boundaries
    """
    M = len(w1)
    N = len(w2)
    if M == 0 or N == 0:
        # Corner case
        return np.sum(w1) + np.sum(w2)
    D = np.inf*np.ones((M, N))
    back = [[[] for j in range(N)] for i in range(M)]
    # Boundary conditions
    for j in range(N):
        D[0, j] = np.abs(w1[0] - np.sum(w2[0:j+1]))
        back[0][j] = [0, j-1]
    for i in range(M):
        D[i, 0] = np.abs(np.sum(w1[0:i+1]) - w2[0])
        back[i][0] = [i-1, 0]
    for i in range(1, M):
        for j in range(1, N):
            for k in range(i):
                for l in range(j):
                    sum1 = np.sum(w1[k+1:i+1])
                    sum2 = np.sum(w2[l+1:j+1])
                    res = D[k, l] + np.abs(sum1-sum2)
                    if res < D[i, j]:
                        D[i, j] = res
                        back[i][j] = [k, l]
    i = M-1
    j = N-1
    path = []
    while i > 0 or j > 0:
        path.append([i, j])
        i, j = back[i][j]
    path.append([0, 0])
    return D[-1, -1], path[::-1]