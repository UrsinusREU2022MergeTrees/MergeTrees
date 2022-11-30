import numpy as np
import matplotlib.pyplot as plt
from mergetree import *
from matching import *

[HORIZ, VERT, DIAG] = [0, 1, 2]

def plot_wasserstein_matching(dgm1, dgm2, matching, labels=["dgm1", "dgm2"],
                              markers=["o", "x"], sizes=[20, 40], colors=["C0", "C1"], 
                              plot_pairs_text=False, ax=None, diag_type=[DIAG]):
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
    diag_type: int
        Type of L1 diagonal plot to do (diagonal/horizontal/vertical)

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
                x, y = b, d
                if diag_type == HORIZ:
                    x, y = d, d
                elif diag_type == DIAG:
                    x, y = (b+d)/2, (b+d)/2
                ax.plot([dgm2[j, 0], x], [dgm2[j, 1], y], "k")
                if plot_pairs_text:
                    x, y = 0.5*(dgm2[j, :] + np.array([x, y]))
                    ax.text(x, y, "{}".format(j), c=colors[1])
            elif j == -1:
                [b, d] = dgm1[i, :]
                x, y = b, b
                if diag_type == HORIZ:
                    x, y = d, d
                elif diag_type == DIAG:
                    x, y = (b+d)/2, (b+d)/2
                ax.plot([dgm1[i, 0], x], [dgm1[i, 1], y], "k")
                if plot_pairs_text:
                    x, y = 0.5*(dgm1[i, :] + np.array([x, y]))
                    ax.text(x, y, "{}".format(i), c=colors[0])
            else:
                ax.plot([dgm1[i, 0], dgm2[j, 0]], [dgm1[i, 1], dgm2[j, 1]], "k")
                if plot_pairs_text:
                    x, y = 0.5*(dgm1[i, :] + dgm2[j, :])
                    if diag_type == VERT or diag_type == DIAG:
                        x += 0.4
                        y -= 0.2
                    if diag_type == HORIZ or diag_type == DIAG:
                        y += 0.3
                        x -= 0.5
                    ax.text(x, y, "({}".format(i), c=colors[0])
                    ax.text(x+0.8, y, ",{})".format(j), c=colors[1])

def plot_delete_move(x_orig, idx1, idx2, h=None, tmax1=1, tmax2=1):
    """
    Show what happens along a vertical trajectory moving 
    critical points to a target height

    Parameters
    ----------
    x_orig: ndarray(N)
        Time series of critical points
    idx1: int
        Index of point that's moving
    idx2: int
        Index of point that's staying still, assumed to be
        either directly to the left or right of idx1
    h: float
        Target height of both points.  If None, make it the height of the second point
    tmax1: float
        How far along its trajectory to move the first point.  0 is beginning, 1 is end
    tmax2: float
        How far along its trajectory to move the second point.  0 is beginning, 1 is end
    """
    N = x_orig.size
    assert(idx1 >= 0)
    assert(idx2 >= 0)
    assert(idx1 < N)
    assert(idx2 < N)
    assert(abs(idx1-idx2) == 1) # Must be adjacent!

    if h is None:
        h = x_orig[idx2]
    
    ## Step 1: Construct time series that results from deleting these two points
    ## as well as the time series that results from moving the mobile point
    idx_orig = np.arange(x_orig.size)

    idxmin = min(idx1, idx2)
    idx_del = np.concatenate((np.arange(idxmin), np.arange(idxmin+2, x_orig.size)))
    x_del = np.concatenate((x_orig[0:idxmin], x_orig[idxmin+2::]))

    x_final = np.array(x_orig)
    for idx, tmax in zip([idx1, idx2], [tmax1, tmax2]):
        x_final[idx] = tmax*h + (1-tmax)*x_orig[idx]
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
    ax_orig.set_title("Height Difference: {:.2f}".format(np.abs(x_orig[idx1]-x_orig[idx2])))
    
    ax_final = plt.subplot(232)
    ax_final.plot(x_final, c='C1')
    ax_final.scatter(np.arange(len(x_final)), x_final, c='C1')
    ax_final.set_title("Final Time Series")

    ax_del = plt.subplot(234)
    ax_del.plot(idx_del, x_del, c='C2')
    ax_del.scatter(idx_del, x_del, c='C2')
    ax_del.set_title("Deleted Time Series")

    ax_wass_orig_final = plt.subplot(233)
    ax_wass_orig_final.set_title("Wass Distance: {:.2f}".format(dwass_orig_final))
    plot_wasserstein_matching(MTOrig.PD, MTFinal.PD, match_orig_final, ax=ax_wass_orig_final, colors=["C0", "C1"], plot_pairs_text=True)
    
    ax_wass_orig_del = plt.subplot(235)
    plot_wasserstein_matching(MTOrig.PD, MTDel.PD, match_orig_del, ax=ax_wass_orig_del, colors=["C0", "C2"], plot_pairs_text=True)
    ax_wass_orig_del.set_title("Wass Distance: {:.2f}".format(dwass_orig_del))
    
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


def animate_delete_moves(x_orig, idx1, idx2, h=None, tmin=0, tmax=1, n_frames=100, prefix=""):
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
    h: float
        Target height of both points.  If None, make it the height of the second point
    tmin: float
        Starting time
    tmax: float
        Ending time
    n_frames: int
        Number of frames in the animation
    prefix: string
        Prefix of filename of saved animation frames
    """
    idx = 0
    for t in np.linspace(tmin, tmax, n_frames//2):
        plt.clf()
        plot_delete_move(x_orig, idx1, idx2, h, tmax1=t, tmax2=0)
        plt.savefig("{}{}.png".format(prefix, idx))
        idx += 1
    for t in np.linspace(tmin, tmax, n_frames//2):
        plt.clf()
        plot_delete_move(x_orig, idx1, idx2, h, tmax1=1, tmax2=t)
        plt.savefig("{}{}.png".format(prefix, idx))
        idx += 1



def plot_delete_figure(x_orig, idx):
    """
    Show moving a min up and then moving a max down

    Parameters
    ----------
    x_orig: ndarray(N)
        Time series of critical points
    idx: int
        Index of first point in an adjacent pair to delete
    """
    h = 0.5*(x_orig[idx-1]+x_orig[idx+2])
    
    ## Step 1: Construct time series that result from moving the first point
    ## and then from moving the second point after that
    x_mid = np.array(x_orig)
    x_mid[idx] = h

    x_end = np.array(x_mid)
    x_end[idx+1] = h

    ## Step 2: Compute merge trees and persistence diagrams of 
    ## the original time series, the time series with delete points,
    ## and the final time series, as well as Wasserstein distances between them
    MTOrig = MergeTree(x_orig)
    MTMid = MergeTree(x_mid)
    MTEnd = MergeTree(x_end)
    dwass_orig_mid, match_orig_mid = wasserstein(MTOrig.PD, MTMid.PD, matching=True)
    dwass_mid_end, match_mid_end = wasserstein(MTMid.PD, MTEnd.PD, matching=True)
    
    ## Step 3: Plot the time series and the Wasserstein matchings
    """
    Original TS     Mid TS     Original->Mid TS Wass    

                    End TS     Mid->End TS Wass
    """
    mn, mx = np.min(x_orig), np.max(x_orig)
    rg = mx-mn
    lims = [mn-1, mx+1]

    ax_orig = plt.subplot(231)
    ax_orig.plot(x_orig, c='C0')
    ax_orig.scatter(np.arange(len(x_orig)), x_orig, c='C0')
    ax_orig.plot([idx, idx+1], x_orig[idx:idx+2], c='C3', linewidth=3)
    ax_orig.set_title("Height Difference: {}".format(np.abs(x_orig[idx]-x_orig[idx+1])))
    
    ax_mid = plt.subplot(232)
    ax_mid.plot(x_mid, c='C1')
    ax_mid.scatter(np.arange(len(x_mid)), x_mid, c='C1')
    ax_mid.set_title("Moving Min Up")
    ax_mid.arrow(idx, x_orig[idx], 0, h-x_orig[idx]-2, color='C1', head_width=0.6, head_length=1)

    ax_end = plt.subplot(235)
    ax_end.plot(x_end, c='C2')
    ax_end.scatter(np.arange(x_end.size), x_end, c='C2')
    ax_end.set_title("Moving Max Down")
    ax_end.arrow(idx, x_orig[idx], 0, h-x_orig[idx]-2, color='C1', head_width=0.6, head_length=1)
    ax_end.arrow(idx+1, x_orig[idx+1], 0, h-x_orig[idx+1]+1, color='C2', head_width=0.6, head_length=1)

    ax_wass_orig_mid = plt.subplot(233)
    ax_wass_orig_mid.set_title("Wass Original -> Min Up: {}".format(int(dwass_orig_mid)))
    plot_wasserstein_matching(MTOrig.PD, MTMid.PD, match_orig_mid, ax=ax_wass_orig_mid, colors=["C0", "C1"], plot_pairs_text=True, diag_type=HORIZ, labels=["Original", "Min Up"])
    
    ax_wass_mid_end = plt.subplot(236)
    plot_wasserstein_matching(MTMid.PD, MTEnd.PD, match_mid_end, ax=ax_wass_mid_end, colors=["C1", "C2"], plot_pairs_text=True, diag_type=VERT, labels=["Min Up", "Max Down"])
    ax_wass_mid_end.set_title("Wass Min Up -> Max Down: {}".format(int(dwass_mid_end)))
    
    ax_del = plt.subplot(234)
    idx_del = np.arange(x_orig.size)
    idx_del = np.concatenate((idx_del[0:idx], idx_del[idx+2::]))
    x_del = np.concatenate((x_orig[0:idx], x_orig[idx+2::]))
    ax_del.plot(idx_del, x_del, c='C4')
    ax_del.scatter(idx_del, x_del, c='C4')
    plt.title("Adjacent Pair Deleted")

    for ax in [ax_orig, ax_mid, ax_end, ax_del]:
        ax.set_xticks([])
    for ax in [ax_orig, ax_mid, ax_end, ax_del, ax_wass_orig_mid, ax_wass_mid_end]:
        ax.set_ylim(lims)
        ax.set_yticks(np.arange(0, np.max(x_orig)+1))
        ax.grid(True, linestyle='--')
    for ax in [ax_wass_orig_mid, ax_wass_mid_end]:
        ax.set_xlim(lims)
        ax.set_xticks(np.arange(0, np.max(x_orig)+1))

    ## Step 4: Plot labels on original time series corresponding to
    ## indices in the persistence diagrams
    for c, (ax, x_this, PDIdx) in enumerate(zip([ax_orig, ax_mid, ax_end],
                        [x_orig, x_mid, x_end],
                        [MTOrig.PDIdx, MTMid.PDIdx, MTEnd.PDIdx])):
        c="C{}".format(c)
        xb2pidx = {x:i for i, x in enumerate(PDIdx[:, 0])}
        d2bx = {d:b for [b, d] in PDIdx}
        for i, v in enumerate(x_this):
            if i in xb2pidx:
                ax.text(i, v+0.01*rg, "{}".format(xb2pidx[i]), c=c)
            elif i in d2bx and d2bx[i] in xb2pidx:
                ax.text(i, v+0.02*rg, "{}".format(xb2pidx[d2bx[i]]), c=c)


def plot_dope_matching(x, y, xc, xs, xc_idx, yc, ys, yc_idx, cost, xdel, ydel, xname="x", yname="y", xcolor="C0", ycolor="C1", plot_matches=True, plot_verified=True, tsname="Original"):
    """
    Create a plot of a particular dope matching, showing x/y deletions and
    matchings
    """
    ## Step 1: Show critical points deleted from x
    lims = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
    rg = lims[1] - lims[0]
    lims = [lims[0]-0.1*rg, lims[1]+0.1*rg]
    xdel_plot = plt.subplot(334)
    xdel_plot.plot(xc, c=xcolor)
    ilast = 0
    xnew = []
    xnew_idx = []
    xidx = np.arange(xc.size)
    xcost = 0
    for rg in xdel:
        xcost += np.sum(xc[rg[0]:rg[1]]*xs[rg[0]:rg[1]])
        xdel_plot.scatter(np.arange(rg[0], rg[1]), xc[rg[0]:rg[1]], c='C3', marker='x', zorder=100)
        xnew = np.concatenate((xnew, xc[ilast:rg[0]]))
        xnew_idx = np.concatenate((xnew_idx, xidx[ilast:rg[0]]))
        ilast = rg[1]
    xnew = np.concatenate((xnew, xc[ilast::]))
    xnew_idx = np.array(np.concatenate((xnew_idx, xidx[ilast::])), dtype=int)
    xdel_plot.set_title("{} critical points\nDeletion Cost={:.3f}".format(xname, xcost))
    xdel_plot.set_ylim(lims)

    ## Step 2: Show critical points deleted from y
    ydel_plot = plt.subplot(335)
    ydel_plot.plot(yc, c=ycolor)
    ilast = 0
    ynew = []
    ynew_idx = []
    yidx = np.arange(yc.size)
    ycost = 0
    for rg in ydel:
        ycost += np.sum(yc[rg[0]:rg[1]]*ys[rg[0]:rg[1]])
        ydel_plot.scatter(np.arange(rg[0], rg[1]), yc[rg[0]:rg[1]], c='C3', marker='x', zorder=100)
        ynew = np.concatenate((ynew, yc[ilast:rg[0]]))
        ynew_idx = np.concatenate((ynew_idx, yidx[ilast:rg[0]]))
        ilast = rg[1]
    ynew = np.concatenate((ynew, yc[ilast::]))
    ynew_idx = np.array(np.concatenate((ynew_idx, yidx[ilast::])), dtype=int)
    ydel_plot.set_title("{} critical points\nDeletion Cost={:.3f}".format(yname, ycost))
    ydel_plot.set_ylim(lims)

    ## Step 3: Show aligned points
    match_plot = plt.subplot(336)
    l1cost = np.sum(np.abs(xnew-ynew))
    match_plot.plot(xnew, c=xcolor)
    match_plot.plot(ynew, c=ycolor, linestyle='--')
    if plot_matches:
        for i, (xi, yi) in enumerate(zip(xnew, ynew)):
            plt.plot([i, i], [xi, yi], c='k', linestyle='--')
            plt.scatter([i, i], [xi, yi], s=20, c='k', zorder=100)
    match_plot.set_title("L1 Matched Points\n L1 Cost={:.3f}".format(l1cost))
    match_plot.set_ylim(lims)

    ## Step 4: Plot original time series
    plt.subplot2grid((3, 3), (0, 0), colspan=3)
    plt.plot(x, c=xcolor)
    plt.plot(y, c=ycolor)
    plt.ylim(lims)
    title = "{} Time Series, Dope Cost: {:.3f}".format(tsname, cost)
    if plot_verified:
        title += ", Verified Cost {:.3f}".format(xcost + ycost + l1cost)
    plt.title(title)
    plt.legend([xname, yname])
    if plot_matches:
        for (xidx, yidx) in zip(xnew_idx, ynew_idx):
            xidx = xc_idx[xidx]
            yidx = yc_idx[yidx]
            plt.plot([xidx, yidx], [x[xidx], y[yidx]], c='k', linestyle='--')
            plt.scatter([xidx, yidx], [x[xidx], y[yidx]], s=20, c='k', zorder=100)
    plt.tight_layout()