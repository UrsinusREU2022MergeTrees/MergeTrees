import numpy as np
from numba import jit
from numba.types import int_, float32
from mergetree import *

@jit("Tuple((f4,i8[:,:],i8[:,:]))(f8[:],f8[:])", nopython=True)
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
    xcosts = np.zeros(x.size+1, float32)
    xcosts[1::] = np.cumsum(x*xs)
    ycosts = np.zeros(y.size+1, float32)
    ycosts[1::] = np.cumsum(y*ys)
    if M == 0 or N == 0:
        # Corner case
        return xcosts[-1] + ycosts[-1], np.zeros((0,0), int_), np.zeros((0,0), int_)

    ## Step 2: Setup data structures
    # Dynamic programming matrix
    D = np.inf*np.ones((M+1, N+1), float32)
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
            # This should only be done if the last points are both mins or both maxes
            if xs[i-1] == ys[i-1]:
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
    path.reverse()

    ## Step 5: Extract and merge deleted chunk indices
    xdel = []
    ydel = []
    for i in range(len(path)-1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        diff = p2 - p1
        if diff[0] > 1:
            xnext = [p1[0], p2[0]]
            if len(xdel) > 0 and xdel[-1][1] == xnext[0]:
                # Merge with last chunk
                xdel[-1][1] = xnext[1]
            else:
                xdel.append(xnext)
        if diff[1] > 1:
            ynext = [p1[1], p2[1]]
            if len(ydel) > 0 and ydel[-1][1] == ynext[0]:
                # Merge with last chunk
                ydel[-1][1] = ynext[1]
            else:
                ydel.append(ynext)
    if len(xdel) == 0:
        xdel = np.zeros((0,0), int_)
    else:
        xdel = np.array(xdel, int_)
    if len(ydel) == 0:
        ydel = np.zeros((0,0), int_)
    else:
        ydel = np.array(ydel, int_)
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



def wasserstein(dgm1, dgm2, matching=False):
    from scipy import optimize
    """
    Perform the L1 Wasserstein distance matching between persistence diagrams.
    Assumes first two columns of dgm1 and dgm2 are the coordinates of the persistence
    points, but allows for other coordinate columns (which are ignored in
    diagonal matching).

    See the `distances` notebook for an example of how to use this.

    Parameters
    ------------

    dgm1: Mx(>=2) 
        array of birth/death pairs for PD 1
    dgm2: Nx(>=2) 
        array of birth/death paris for PD 2
    matching: bool, default False
        if True, return matching information and cross-similarity matrix

    Returns 
    ---------

    d: float
        Wasserstein distance between dgm1 and dgm2
    (matching, D): Only returns if `matching=True`
        (tuples of matched indices, (N+M)x(N+M) cross-similarity matrix)

    """
    from warnings import warn
    S = np.array(dgm1)
    M = min(S.shape[0], S.size)
    if S.size > 0:
        S = S[np.isfinite(S[:, 1]), :]
        if S.shape[0] < M:
            warn(
                "dgm1 has points with non-finite death times;"+
                "ignoring those points"
            )
            M = S.shape[0]
    T = np.array(dgm2)
    N = min(T.shape[0], T.size)
    if T.size > 0:
        T = T[np.isfinite(T[:, 1]), :]
        if T.shape[0] < N:
            warn(
                "dgm2 has points with non-finite death times;"+
                "ignoring those points"
            )
            N = T.shape[0]

    if M == 0:
        S = np.array([[0, 0]])
        M = 1
    if N == 0:
        T = np.array([[0, 0]])
        N = 1
    # Compute CSM between S and dgm2, including points on diagonal
    DUL  = np.abs(S[:, 0][:, None] - T[:, 0][None, :])
    DUL += np.abs(S[:, 1][:, None] - T[:, 1][None, :])

    # Put diagonal elements into the matrix
    D = np.zeros((M+N, M+N))
    np.fill_diagonal(D, 0)
    D[0:M, 0:N] = DUL
    UR = np.inf*np.ones((M, M))
    np.fill_diagonal(UR, S[:, 1]-S[:, 0])
    D[0:M, N:N+M] = UR
    UL = np.inf*np.ones((N, N))
    np.fill_diagonal(UL, T[:, 1]-T[:, 0])
    D[M:N+M, 0:N] = UL

    # Step 2: Run the hungarian algorithm
    matchi, matchj = optimize.linear_sum_assignment(D)
    matchdist = np.sum(D[matchi, matchj])

    if matching:
        matchidx = [(i, j) for i, j in zip(matchi, matchj)]
        ret = np.zeros((len(matchidx), 3))
        ret[:, 0:2] = np.array(matchidx)
        ret[:, 2] = D[matchi, matchj]
        # Indicate diagonally matched points
        ret[ret[:, 0] >= M, 0] = -1
        ret[ret[:, 1] >= N, 1] = -1
        # Exclude diagonal to diagonal
        ret = ret[ret[:, 0] + ret[:, 1] != -2, :] 
        return matchdist, ret

    return matchdist



def wasserstein_matching(dgm1, dgm2, matching, labels=["dgm1", "dgm2"], ax=None):
    """ Visualize Wasserstein matching between two diagrams

    Parameters
    ===========

    dgm1: array
        A diagram
    dgm2: array
        A diagram
    matching: ndarray(Mx+Nx, 3)
        A list of correspondences in an optimal matching, as well as their distance, where:
        * First column is index of point in first persistence diagram, or -1 if diagonal
        * Second column is index of point in second persistence diagram, or -1 if diagonal
        * Third column is the distance of each matching
    labels: list of strings
        names of diagrams for legend. Default = ["dgm1", "dgm2"], 
    ax: matplotlib Axis object
        For plotting on a particular axis.

    Examples
    ==========

    bn_matching, matchidx = wasserstien(A_h1, B_h1, matching=True)
    wasserstein_matching(A_h1, B_h1, matchidx)

    """
    ax = ax or plt.gca()
    if dgm1.size == 0:
        dgm1 = np.array([[0, 0]])
    if dgm2.size == 0:
        dgm2 = np.array([[0, 0]])
    for [i, j, d] in matching:
        i = int(i)
        j = int(j)
        if i != -1 or j != -1: # At least one point is a non-diagonal point
            if i == -1:
                [b, d] = dgm2[j, :]
                plt.plot([dgm2[j, 0], (b+d)/2], [dgm2[j, 1], (b+d)/2], "g")
            elif j == -1:
                [b, d] = dgm1[i, :]
                ax.plot([dgm1[i, 0], (b+d)/2], [dgm1[i, 1], (b+d)/2], "g")
            else:
                ax.plot([dgm1[i, 0], dgm2[j, 0]], [dgm1[i, 1], dgm2[j, 1]], "g")

    plot_diagrams([dgm1, dgm2], labels=labels, ax=ax)