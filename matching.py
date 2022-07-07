import numpy as np
#from numba import jit
from mergetree import *

def dpw(x, y):
    """
    Compute dynamic persistence warping (DPW) between two time series

    Parameters
    ----------
    x: ndarray(M)
        First time series
    y: ndarray(N)
        Second time series
    
    Returns
    -------
    float: optimal distance,
    list of [int, int]: Ranges to delete in x critical time series
    list of [int, int]: Ranges to delete in y critical time series
    """
    ## Step 1: Setup critical point time series and costs
    x, xs = get_crit_timeseries(x, circular=True)
    y, ys = get_crit_timeseries(y, circular=True)
    M = x.size
    N = y.size
    # Use cumulative sums for quick deletion cost lookup
    xcosts = np.concatenate(([0], np.cumsum(x*xs)))
    ycosts = np.concatenate(([0], np.cumsum(y*ys)))
    if M == 0 or N == 0:
        # Corner case
        return xcosts[-1] + ycosts[-1]

    ## Step 2: Setup data structures
    # Dynamic programming matrix
    D = np.inf*np.ones((M+1, N+1))
    D[0, 0] = 0
    # Backtracing matrix; each entry is sizes of chunks deleted from each time series
    back = [[[0, 0] for j in range(N+1)] for i in range(M+1)]
    # Boundary conditions
    D[0, 2::2] = ycosts[2::2]
    D[2::2, 0] = xcosts[2::2]
    for j in range(2, N+1, 2):
        back[0][j] = [0, j]
    for i in range(2, M+1, 2):
        back[i][0] = [i, 0]
    
    ## Step 3: Do dynamic programming
    for i in range(1, M+1):
        for j in range(1, N+1):
            # First try matching the last point in the L1 sum
            D[i, j] = np.abs(x[i-1]-y[j-1]) + D[i-1, j-1]
            # Now try all other deletion possibilities
            for k, l in [[0, 2], [2, 0], [2, 2]]:
                if i >= k and j >= l:
                    # Delete chunks from each time series and use
                    # the previously computed subproblem
                    xdel = 0
                    if k > 0:
                        xdel = xcosts[i]-xcosts[i-k]
                    ydel = 0
                    if l > 0:
                        ydel = ycosts[j]-ycosts[j-l]
                    res = D[i-k, j-l] + xdel + ydel
                    if res < D[i, j]:
                        D[i, j] = res
                        back[i][j] = [k, l]
    
    ## Step 4: Do backtracing 
    i = M
    j = N
    path = []
    while i > 0 or j > 0:
        path.append([i, j])
        [di, dj] = back[i][j]
        if di == 0 and dj == 0:
            i -= 1
            j -= 1
        else:
            i -= di
            j -= dj
    path.append([0, 0])

    xdel = []
    ydel = []
    for i in range(len(path)-1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        diff = p1 - p2
        if diff[0] > 1:
            xdel.append([p2[0], p1[0]])
        if diff[1] > 1:
            ydel.append([p2[1], p1[1]])
    xdel.reverse()
    ydel.reverse()

    return D[-1, -1], xdel, ydel



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