import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba.types import int_, float32
from mergetree import *

@jit(nopython=True)
def merge_chunks(path):
    """
    Coalesce the deleted chunks in a path

    Parameters
    ----------
    path: list of [i, j]
        Backtrace edit path

    Returns
    -------
    xdel: list of [i1, i2), each of which is a range to delete from x
    ydel: list of [j1, j2), each of which is a range to delete from y
    """
    xdel = []
    ydel = []
    for i in range(len(path)-1):
        p1 = path[i, :]
        p2 = path[i+1, :]
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
    return xdel, ydel

def get_matched_points(path):
    """
    Get the matched indices in a path

    Parameters
    ----------
    path: ndarray(K, 2, dtype=int)
        Backtrace edit path

    Returns
    -------
    list of [i, j]: matchings
    """
    matching = []
    for i in range(1, len(path)):
        diff = path[i, :] - path[i-1, :]
        if diff[0] == 1 and diff[1] == 1:
            matching.append(path[i-1])
    return matching


@jit("Tuple((f4,i8[:,:]))(f8[:],f8[:],b1)", nopython=True)
def dope_match(x, y, circular=False):
    """
    Compute Dynamic Ordered Persistence Editing (DOPE) between two time series

    Parameters
    ----------
    x: ndarray(M)
        First time series
    y: ndarray(N)
        Second time series
    
    Returns
    -------
    float: optimal distance,
    list of [int, int]: Backtrace path
    """
    ## Step 1: Setup critical point time series and costs
    x, xs, _ = get_crit_timeseries(x, circular=circular)
    y, ys, _ = get_crit_timeseries(y, circular=circular)
    M = x.size
    N = y.size
    # Use cumulative sums for quick deletion cost lookup
    xcosts = np.zeros(x.size+1, float32)
    xcosts[1::] = np.cumsum(x*xs)
    ycosts = np.zeros(y.size+1, float32)
    ycosts[1::] = np.cumsum(y*ys)
    if M == 0 or N == 0:
        # Corner case
        return xcosts[-1] + ycosts[-1], np.zeros((0,0), int_)

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
            if xs[i-1] == ys[j-1]:
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
    path = np.array(path, int_)

    ## Step 5: Extract and merge deleted chunk indices
    return D[-1, -1], path



@jit("Tuple((f4,i8,i8,i8[:,:]))(f8[:],f8[:])", nopython=True)
def circular_dope_match(x, y):
    """
    Compute Dynamic Ordered Persistence Editing (DOPE) between two time series

    Parameters
    ----------
    x: ndarray(M)
        First time series
    y: ndarray(N)
        Second time series
    
    Returns
    -------
    float: optimal distance
    int: Optimal x_shift
    int: Optimal y_shift
    ndarray(K, 2)
        Optimal warping path
    """
    ## Step 1: Setup critical point time series and costs
    xc, xs, _ = get_crit_timeseries(x, circular=True)
    yc, ys, _ = get_crit_timeseries(y, circular=True)
    M = xc.size
    N = yc.size
    if M == 0 or N == 0:
        # Corner case
        return float32(np.sum(xc*xs) + np.sum(yc*ys)), int_(0), int_(0), np.zeros((0,0), int_)
    
    
    xc2 = np.roll(xc, 1) # Make another version of x that's been circularly shifted
    y_shift = 0
    min_cost = np.inf
    min_xshift = 0
    min_yshift = 0
    min_path= np.zeros((0,0), int_)
    for _ in range(N):
        # First do with original x
        cost, path = dope_match(xc, yc, True)
        if cost < min_cost:
            min_cost = cost
            min_xshift = 0
            min_yshift = y_shift
            min_path = path
        # Now redo with x circularly shifted by 1 spot
        cost, path = dope_match(xc2, yc, True)
        if cost < min_cost:
            min_cost = cost
            min_xshift = 1
            min_yshift = y_shift
            min_path = path
        # Shift for the next alignment attempt
        yc = np.roll(yc, 1)
        y_shift += 1
    return min_cost, min_xshift, min_yshift, min_path


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



def bottleneck(dgm1, dgm2, matching=False):
    """
    Perform the Bottleneck distance matching between persistence diagrams.
    Assumes first two columns of S and T are the coordinates of the persistence
    points, but allows for other coordinate columns (which are ignored in
    diagonal matching).
    See the `distances` notebook for an example of how to use this.
    Parameters
    -----------
    dgm1: Mx(>=2) 
        array of birth/death pairs for PD 1
    dgm2: Nx(>=2) 
        array of birth/death paris for PD 2
    matching: bool, default False
        if True, return matching infromation and cross-similarity matrix
    Returns
    --------
    d: float
        bottleneck distance between dgm1 and dgm2
    matching: ndarray(Mx+Nx, 3), Only returns if `matching=True`
        A list of correspondences in an optimal matching, as well as their distance, where:
        * First column is index of point in first persistence diagram, or -1 if diagonal
        * Second column is index of point in second persistence diagram, or -1 if diagonal
        * Third column is the distance of each matching
    """
    from bisect import bisect_left
    from hopcroftkarp import HopcroftKarp
    from warnings import warn

    return_matching = matching

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
    if M == 0 and N == 0:
        if matching:
            return 0, np.array([])
        else:
            return 0
    if M == 0:
        S = np.array([[0, 0]])
        M = 1
    if N == 0:
        T = np.array([[0, 0]])
        N = 1

    # Step 1: Compute CSM between S and T, including points on diagonal
    # L Infinity distance
    Sb, Sd = S[:, 0], S[:, 1]
    Tb, Td = T[:, 0], T[:, 1]
    D1 = np.abs(Sb[:, None] - Tb[None, :])
    D2 = np.abs(Sd[:, None] - Td[None, :])
    DUL = np.maximum(D1, D2)

    # Put diagonal elements into the matrix, being mindful that Linfinity
    # balls meet the diagonal line at a diamond vertex
    D = np.zeros((M + N, M + N))
    # Upper left is Linfinity cross-similarity between two diagrams
    D[0:M, 0:N] = DUL
    # Upper right is diagonal matching of points from S
    UR = np.inf * np.ones((M, M))
    np.fill_diagonal(UR, 0.5 * (S[:, 1] - S[:, 0]))
    D[0:M, N::] = UR
    # Lower left is diagonal matching of points from T
    UL = np.inf * np.ones((N, N))
    np.fill_diagonal(UL, 0.5 * (T[:, 1] - T[:, 0]))
    D[M::, 0:N] = UL
    # Lower right is all 0s by default (remaining diagonals match to diagonals)

    # Step 2: Perform a binary search + Hopcroft Karp to find the
    # bottleneck distance
    ds = np.sort(np.unique(D.flatten()))[0:-1] # Everything but np.inf
    bdist = ds[-1]
    matching = {}
    while len(ds) >= 1:
        idx = 0
        if len(ds) > 1:
            idx = bisect_left(range(ds.size), int(ds.size / 2))
        d = ds[idx]
        graph = {}
        for i in range(D.shape[0]):
            graph["{}".format(i)] = {j for j in range(D.shape[1]) if D[i, j] <= d}
        res = HopcroftKarp(graph).maximum_matching()
        if len(res) == 2 * D.shape[0] and d <= bdist:
            bdist = d
            matching = res
            ds = ds[0:idx]
        else:
            ds = ds[idx + 1::]

    if return_matching:
        matchidx = []
        for i in range(M+N):
            j = matching["{}".format(i)]
            d = D[i, j]
            if i < M:
                if j >= N:
                    j = -1 # Diagonal match from first persistence diagram
            else:
                if j >= N: # Diagonal to diagonal, so don't include this
                    continue
                i = -1
            matchidx.append([i, j, d])
        return bdist, np.array(matchidx)
    else:
        return bdist