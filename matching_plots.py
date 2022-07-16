import numpy as np
import matplotlib.pyplot as plt
from mergetree import *
from matching import *


def plot_wasserstein_matching(dgm1, dgm2, matching, labels=["dgm1", "dgm2"], 
                              markers=["o", "x"], sizes=[20, 40], colors=["C0", "C1"], 
                              plot_pairs_text=False, ax=None):
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
    markers: list of strings
        Marker types to use for the diagrams.  Default = ["o", "x"]
    sizes: list of [float, float]
        Sizes of markers in each diagram.  Default = [20, 100]
    colors: list of [string, string]
        Colors to use for each diagram. Default = ["C0", "C1"]
    plot_pairs_text: bool
        If true, plot text indicating a tuple of the matched pairs
    ax: matplotlib Axis object
        For plotting on a particular axis.

    Examples
    ==========

    bn_matching, matchidx = wasserstien(A_h1, B_h1, matching=True)
    wasserstein_matching(A_h1, B_h1, matchidx)

    """
    ax = ax or plt.gca()
    plot_diagrams([dgm1, dgm2], labels=labels, markers=markers, sizes=sizes, colors=colors, ax=ax)
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
                ax.plot([dgm2[j, 0], (b+d)/2], [dgm2[j, 1], (b+d)/2], "k")
                if plot_pairs_text:
                    x, y = dgm2[j, :]
                    plt.text(x, y, "{}".format(j), c=colors[1])
            elif j == -1:
                [b, d] = dgm1[i, :]
                ax.plot([dgm1[i, 0], (b+d)/2], [dgm1[i, 1], (b+d)/2], "k")
                if plot_pairs_text:
                    x, y = dgm1[i, :]
                    plt.text(x, y, "{}".format(i), c=colors[0])
            else:
                ax.plot([dgm1[i, 0], dgm2[j, 0]], [dgm1[i, 1], dgm2[j, 1]], "k")
                if plot_pairs_text:
                    x, y = 0.5*(dgm1[i, :] + dgm2[j, :]) + np.array([0.2, 0])
                    plt.text(x, y, "({}".format(i), c=colors[0])
                    plt.text(x+0.2, y, ",{})".format(j), c=colors[1])

def plot_delete_move(x_orig, idx1, idx2, tmax=1):
    """
    Show what happens along a vertical trajectory moving a single
    critical point to the height of the one next to it

    Parameters
    ----------
    x_orig: ndarray(N)
        Time series of critical points
    idx1: int
        Index of point that's moving
    idx2: int
        Index of point that's staying still, assumed to be
        either directly to the left or right of idx1
    tmax: float
        How far along its trajectory to move the point.  0 is beginning, 1 is end
    """
    N = x_orig.size
    assert(idx1 >= 0)
    assert(idx2 >= 0)
    assert(idx1 < N)
    assert(idx2 < N)
    assert(abs(idx1-idx2) == 1) # Must be adjacent!
    
    ## Step 1: Construct time series that results from deleting these two points
    ## as well as the time series that results from moving the mobile point
    idx_orig = np.arange(x_orig.size)

    idxmin = min(idx1, idx2)
    idx_del = np.concatenate((np.arange(idxmin), np.arange(idxmin+2, x_orig.size)))
    x_del = np.concatenate((x_orig[0:idxmin], x_orig[idxmin+2::]))

    x_final = np.array(x_orig)
    x_final[idx1] = tmax*(x_final[idx2]) + (1-tmax)*x_final[idx1]
    idx_final = np.arange(x_final.size)

    ## Step 2: Compute merge trees and persistence diagrams of 
    ## the original time series, the time series with delete points,
    ## and the final time series, as well as Wasserstein distances between them
    MTOrig = MergeTree(x_orig)
    MTDel = MergeTree(x_del)
    MTFinal = MergeTree(x_final)
    dwass_orig_final, match_orig_final = wasserstein(MTOrig.PD, MTFinal.PD, matching=True)
    dwass_orig_del, match_orig_del = wasserstein(MTOrig.PD, MTDel.PD, matching=True)
    
    ## Step 3: Plot the time series and the Wasserstein matchings
    """
    Original TS     Final TS     Final TS Wass    

                    Deleted TS   Delete TS Wass
    """
    mn, mx = np.min(x_orig), np.max(x_orig)
    rg = mx-mn
    lims = [mn-0.1*rg, mx+0.1*rg]

    ax_orig = plt.subplot(231)
    ax_orig.plot(x_orig, c='C0')
    ax_orig.scatter(np.arange(len(x_orig)), x_orig, c='C0')
    ax_orig.plot([idxmin, idxmin+1], x_orig[idxmin:idxmin+2], c='C3', linewidth=3)
    ax_orig.set_title("Height Difference: {:.3f}".format(np.abs(x_orig[idx1]-x_orig[idx2])))
    
    ax_final = plt.subplot(232)
    ax_final.plot(x_final, c='C1')
    ax_final.scatter(np.arange(len(x_final)), x_final, c='C1')
    ax_final.set_title("Final Time Series")

    ax_del = plt.subplot(234)
    ax_del.plot(idx_del, x_del, c='C2')
    ax_del.scatter(idx_del, x_del, c='C2')
    ax_del.set_title("Deleted Time Series")

    ax_wass_orig_final = plt.subplot(233)
    ax_wass_orig_final.set_title("Wass Distance: {:.3f}".format(dwass_orig_final))
    plot_wasserstein_matching(MTOrig.PD, MTFinal.PD, match_orig_final, ax=ax_wass_orig_final, colors=["C0", "C1"], plot_pairs_text=True)
    
    ax_wass_orig_del = plt.subplot(235)
    plot_wasserstein_matching(MTOrig.PD, MTDel.PD, match_orig_del, ax=ax_wass_orig_del, colors=["C0", "C2"], plot_pairs_text=True)
    ax_wass_orig_del.set_title("Wass Distance: {:.3f}".format(dwass_orig_del))
    
    for ax in [ax_orig, ax_final, ax_del]:
        ax.set_ylim(lims)
    for ax in [ax_wass_orig_final, ax_wass_orig_del]:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims, lims, linestyle='--')

    ## Step 4: Plot labels on original time series corresponding to
    ## indices in the persistence diagrams
    for c, (ax, idx_this, x_this, PDIdx) in enumerate(zip([ax_orig, ax_final, ax_del], 
                        [idx_orig, idx_final, idx_del],
                        [x_orig, x_final, x_del],
                        [MTOrig.PDIdx, MTFinal.PDIdx, MTDel.PDIdx])):
        c="C{}".format(c)
        xb2pidx = {x:i for i, x in enumerate(PDIdx[:, 0])}
        d2bx = {d:b for [b, d] in PDIdx}
        for i, v in enumerate(x_this):
            if i in xb2pidx:
                ax.text(idx_this[i], v+0.01*rg, "{}".format(xb2pidx[i]), c=c)
            elif i in d2bx and d2bx[i] in xb2pidx:
                ax.text(idx_this[i], v+0.02*rg, "{}".format(xb2pidx[d2bx[i]]), c=c)


def animate_delete_moves(x_orig, idx1, idx2, tmin=0, tmax=1, n_frames=100, prefix=""):
    """
    Show what happens along a vertical trajectory moving a single
    critical point to the height of the one next to it

    Parameters
    ----------
    x_orig: ndarray(N)
        Time series of critical points
    idx1: int
        Index of point that's moving
    idx2: int
        Index of point that's staying still, assumed to be
        either directly to the left or right of idx1
    tmin: float
        Starting time
    tmax: float
        Ending time
    n_frames: int
        Number of frames in the animation
    prefix: string
        Prefix of filename of saved animation frames
    """
    for i, t in enumerate(np.linspace(tmin, tmax, n_frames)):
        plt.clf()
        plot_delete_move(x_orig, idx1, idx2, tmax=t)
        plt.savefig("{}{}.png".format(prefix, i))



def plot_dope_matching(x, y, xc, xs, yc, ys, cost, xdel, ydel, circular):
    """
    Create a plot of a particular dope matching, showing x/y deletions and
    matchings, and then show the optimal Wasserstein matching between 
    persistence diagrams, as well as the best inferred Wasserstein matching
    from the dope matching.  Take care to associate indices of persistence
    pairs with the time series
    """
    ## Step 1: Show critical points deleted from x
    xdel_plot = plt.subplot(334)
    plt.plot(xc)
    ilast = 0
    xnew = []
    xnew_idx = []
    xidx = np.arange(xc.size)
    xcost = 0
    for rg in xdel:
        xcost += np.sum(xc[rg[0]:rg[1]]*xs[rg[0]:rg[1]])
        plt.scatter(np.arange(rg[0], rg[1]), xc[rg[0]:rg[1]], c='C3')
        xnew = np.concatenate((xnew, xc[ilast:rg[0]]))
        xnew_idx = np.concatenate((xnew_idx, xidx[ilast:rg[0]]))
        ilast = rg[1]
    xnew = np.concatenate((xnew, xc[ilast::]))
    xnew_idx = np.array(np.concatenate((xnew_idx, xidx[ilast::])), dtype=int)
    plt.title("x critical points, Del Cost={:.3f}".format(xcost))

    ## Step 2: Show critical points deleted from y
    ydel_plot = plt.subplot(335)
    plt.plot(yc, c='C1')
    ilast = 0
    ynew = []
    ynew_idx = []
    yidx = np.arange(yc.size)
    ycost = 0
    for rg in ydel:
        ycost += np.sum(yc[rg[0]:rg[1]]*ys[rg[0]:rg[1]])
        plt.scatter(np.arange(rg[0], rg[1]), yc[rg[0]:rg[1]], c='C3')
        ynew = np.concatenate((ynew, yc[ilast:rg[0]]))
        ynew_idx = np.concatenate((ynew_idx, yidx[ilast:rg[0]]))
        ilast = rg[1]
    ynew = np.concatenate((ynew, yc[ilast::]))
    ynew_idx = np.array(np.concatenate((ynew_idx, yidx[ilast::])), dtype=int)
    plt.title("y critical points, Del Cost={:.3f}".format(ycost))

    ## Step 3: Show aligned points
    match_plot = plt.subplot(336)
    l1cost = np.sum(np.abs(xnew-ynew))
    plt.plot(xnew)
    plt.plot(ynew, linestyle='--')
    plt.title("L1 Aligned Points, Cost={:.3f}".format(l1cost))

    ## Step 4: Plot original time series
    plt.subplot2grid((3, 3), (0, 0), colspan=3)
    plt.plot(x)
    plt.plot(y, c='C1')
    plt.title("Original Time Series, Computed cost: {:.3f}, Verified Cost {:.3f}".format(cost, xcost + ycost + l1cost))
    plt.legend(["x", "y"])

    ## Step 5: Try to construct wasserstein matching from dope matching
    MTx = MergeTree()
    MTx.init_from_timeseries(xc, circular=circular)
    MTy = MergeTree()
    MTy.init_from_timeseries(yc, circular=circular)

    xb2pidx = {x:i for i, x in enumerate(MTx.PDIdx[:, 0])}
    yb2pidx = {y:j for j, y in enumerate(MTy.PDIdx[:, 0])}
    b2dx = {b:d for [b, d] in MTx.PDIdx}
    d2bx = {d:b for [b, d] in MTx.PDIdx}
    b2dy = {b:d for [b, d] in MTy.PDIdx}
    d2by = {d:b for [b, d] in MTy.PDIdx}

    ## Look at what was matched
    xnew_idx_idx = {x:idx for idx, x in enumerate(xnew_idx)}
    ynew_idx_idx = {y:idx for idx, y in enumerate(ynew_idx)}
    mymatching = []
    for i, (x, y) in enumerate(zip(xnew_idx, ynew_idx)):
        if x in xb2pidx:
            match_plot.text(i-0.4, xnew[i], "{}".format(xb2pidx[x]), c='C0')
        elif x in d2bx and d2bx[x] in xb2pidx:
            match_plot.text(i-0.4, xnew[i], "{}".format(xb2pidx[d2bx[x]]), c='C0')
        else:
            match_plot.text(i-0.4, xnew[i], "N", c='C0')
        if y in yb2pidx:
            match_plot.text(i+0.2, ynew[i], "{}".format(yb2pidx[y]), c='C1')
        elif y in d2by and d2by[y] in yb2pidx:
            match_plot.text(i+0.2, ynew[i], "{}".format(yb2pidx[d2by[y]]), c='C1')
        else:
            match_plot.text(i+0.2, ynew[i], "N", c='C1')
        bx, dx = None, None
        by, dy = None, None
        all_found = False
        if xs[x] == -1: 
            # Local min
            bx = x
            by = y
            # Find paired death and check coordinates
            if x in b2dx and b2dx[x] in xnew_idx_idx:
                dxidx = xnew_idx_idx[b2dx[x]]
                if y in b2dy and b2dy[y] in ynew_idx_idx:
                    if dxidx == ynew_idx_idx[b2dy[y]]:
                        dx = b2dx[x]
                        dy = b2dy[y]
                        all_found = True
        else:
            # Local max
            dx = x
            dy = y
            # Find paired birth and check coordinates
            if x in d2bx and d2bx[x] in xnew_idx_idx:
                bxidx = xnew_idx_idx[d2bx[x]]
                if y in d2by and d2by[y] in ynew_idx_idx:
                    if bxidx == ynew_idx_idx[d2by[y]]:
                        bx = d2bx[x]
                        by = d2by[y]
                        all_found = True
            
        if all_found:
            if xs[x] == -1:
                mymatching.append([xb2pidx[bx], yb2pidx[by], np.abs(xc[dx]-xc[bx])+np.abs(yc[dy]-yc[by])])
        else:
            match_plot.scatter([i, i], [xc[x], yc[y]], s=100, marker='x', c='r')
    
    for x in xidx:
        if x in xb2pidx:
            xdel_plot.text(x+0.3, xc[x], "{}".format(xb2pidx[x]), c='C0')
        elif x in d2bx and d2bx[x] in xb2pidx:
            xdel_plot.text(x+0.3, xc[x], "{}".format(xb2pidx[d2bx[x]]), c='C0')
        else:
            xdel_plot.text(x+0.3, xc[x], "N", c='C0')
    all_xdel = np.array([])
    if len(xdel):
        all_xdel = np.concatenate(tuple([np.arange(rg[0], rg[1]) for rg in xdel]))
    for rg in xdel:
        xchunk = xidx[rg[0]:rg[1]]
        for x in xchunk:
            found = False
            if x in b2dx:
                if b2dx[x] in all_xdel:
                    mymatching.append([xb2pidx[x], -1, b2dx[x]-x])
                    found = True
            elif x in d2bx:
                if d2bx[x] in all_xdel:
                    found=True
            if not found:
                xdel_plot.scatter([x], [xc[x]], s=100, marker='x', c='r')
    
    for y in yidx:
        if y in yb2pidx:
            ydel_plot.text(y+0.3, yc[y], "{}".format(yb2pidx[y]), c='C1')
        elif y in d2by and d2by[y] in yb2pidx:
            ydel_plot.text(y+0.3, yc[y], "{}".format(yb2pidx[d2by[y]]), c='C1')
        else:
            ydel_plot.text(y+0.3, yc[y], "N", c='C1')
    all_ydel = np.array([])
    if len(ydel) > 0:
        all_ydel = np.concatenate(tuple([np.arange(rg[0], rg[1]) for rg in ydel]))
    for rg in ydel:
        ychunk = yidx[rg[0]:rg[1]]
        for y in ychunk:
            found = False
            if y in b2dy:
                if b2dy[y] in all_ydel:
                    mymatching.append([-1, yb2pidx[y], b2dy[y]-y])
                    found = True
            elif y in d2by:
                if d2by[y] in all_ydel:
                    found = True
            if not found:
                ydel_plot.scatter([y], [yc[y]], s=100, marker='x', c='r')
        
    dist, wassmatching = wasserstein(MTx.PD, MTy.PD, True)
    plt.subplot(337)
    plot_wasserstein_matching(MTx.PD, MTy.PD, wassmatching)
    for i, [b, d] in enumerate(MTx.PD):
        plt.text(b-0.2, d, "{}".format(i), c='C0')
    for i, [b, d] in enumerate(MTy.PD):
        plt.text(b-0.2, d, "{}".format(i), c='C1')
    plt.title("Wasserstein Matching, Cost={:.3f}".format(dist))
    plt.subplot(338)
    plot_wasserstein_matching(MTx.PD, MTy.PD, mymatching)
    for i, [b, d] in enumerate(MTx.PD):
        plt.text(b-0.2, d, "{}".format(i), c='C0')
    for i, [b, d] in enumerate(MTy.PD):
        plt.text(b-0.2, d, "{}".format(i), c='C1')
    plt.title("Partial DOPE->Wasserstein Matching")
