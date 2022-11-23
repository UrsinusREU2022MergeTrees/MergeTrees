"""
Implement simple polynomial interpolation to help draw smooth curves
on the merge trees
"""
import numpy as np
import matplotlib.pyplot as plt


def check_triangle_inequality(D, do_plot=True):
    """
    Exhaustively check the triangle inequality on all
    triples of points in a dissimilarity matrix

    Parameters
    ----------
    D: ndarray(N, N)
        Dissimilarity matrix
    do_plot: boolean
        Whether to plot a histogram of the discrepancies in violated triples
    
    Returns
    -------
    ndarray(M)
        A list of all failed discrepancies in violated triples
    """
    N = D.shape[0]
    correct = 0
    failed = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if D[i, k] + D[k, j] >= D[i, j]:
                    correct += 1
                else:
                    failed.append([i, j, k, D[i, j], D[i, k]+D[k, j]])
    failed = np.array(failed)
    if do_plot:
        if failed.size > 0:
            plt.hist((failed[:, -2]-failed[:, -1])/failed[:, -2])
            plt.xlabel("(D(x, y) - (D(x, z) + D(z, y))) / D(x, y)")
            plt.ylabel("Counts")
        plt.title("{:.3f} % Failed".format(100*len(failed)/(correct+len(failed))))
    return failed

def znormalize(x):
    """
    Remove the NaN values from a time series and znormalize what's left

    Parameters
    ----------
    x: ndarray(N)
        Input Time series
    
    Returns
    -------
    y: ndarray(M <= N)
        Z-normalized time series without NaNs
    """
    x = x[~np.isnan(x)]
    x = x - np.mean(x)
    return x/np.std(x)

def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    markers=None,
    sizes=None,
    colors=None,
    colormap="default",
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    equal=True,
    legend=True,
    show=False,
    ax=None
):
    """A helper function to plot persistence diagrams. 
    Parameters
    ----------
    diagrams: ndarray (n_pairs, 2) or list of diagrams
        A diagram or list of diagrams. If diagram is a list of diagrams, 
        then plot all on the same plot using different colors.
    plot_only: list of numeric
        If specified, an array of only the diagrams that should be plotted.
    title: string, default is None
        If title is defined, add it as title of the plot.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing 
        multiple persistence diagrams.
    labels: string or list of strings
        Legend labels for each diagram. 
        If none are specified, we use H_0, H_1, H_2,... by default.
    markers: string or list of strings
        Markers for each diagram
        If none are specified, we use dots by default.
    sizes: int or list of ints
        Sizes of each marker
        If none are specified, use 20 by default
    colors: string or list of strings
        Colors for each diagram
        If none are specified, use the default sequence from matplotlib
    colormap: string, default is 'default'
        Any of matplotlib color palettes. 
        Some options are 'default', 'seaborn', 'sequential'. 
        See all available styles with
        .. code:: python
            import matplotlib as mpl
            print(mpl.styles.available)
    ax_color: any valid matplotlib color type. 
        See [https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html) for complete API.
    diagonal: bool, default is True
        Plot the diagonal x=y line.
    lifetime: bool, default is False. If True, diagonal is turned to False.
        Plot life time of each point instead of birth and death. 
        Essentially, visualize (x, y-x).
    equal: bool, default is True.  If True, plot axes equal
    legend: bool, default is True
        If true, show the legend.
    show: bool, default is False
        Call plt.show() after plotting. If you are using self.plot() as part 
        of a subplot, set show=False and call plt.show() only once at the end.
    """

    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = ["$H_{{{}}}$".format(i) for i , _ in enumerate(diagrams)]
    
    if markers is None:
        markers = ["o"]*len(diagrams)
    
    if sizes is None:
        sizes = [20]*len(diagrams)

    if colors is None:
        colors = ["C{}".format(i) for i in range(len(diagrams))]

    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)
    
    if not isinstance(markers, list):
        markers = [markers]*len(diagrams)
    
    if not isinstance(sizes, list):
        sizes = [sizes]*len(diagrams)
    
    if not isinstance(colors, list):
        colors = [colors]*len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label, marker, size, color in zip(diagrams, labels, markers, sizes, colors):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, c=color, label=label, marker=marker)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    if equal:
        ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="lower right")

    if show is True:
        plt.show()

def get_spline(ys, n_subdivide=10):
    """
    Compute a cublic spline that goes through a time series
    
    Parameters
    ----------
    ys: ndarray(N)
        Time series
    n_subdivide: int
        Number of samples to include between each adjacent pair of time series points

    Return
    ------
    ndarray(N*n_subdivide)
        Samples on spline
    """
    ys = np.array(ys)
    xs = np.arange(ys.size)
    n = len(ys)-1
    a = np.array(ys)
    b = np.zeros(n)
    d = np.zeros(n)
    h = xs[1::]-xs[0:-1]
    alpha = np.zeros(n)
    for i in range(1, n):
        alpha[i] = 3*(a[i+1]-a[i])/h[i] - 3*(a[i]-a[i-1])/h[i-1]
    c = np.zeros(n+1)
    l = np.zeros(n+1)
    mu = np.zeros(n+1)
    z = np.zeros(n+1)
    l[0] = 1
    for i in range(1, n):
        l[i] = 2*(xs[i+1]-xs[i-1])-h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i]-h[i-1]*z[i-1])/l[i]
    l[n] = 1
    for j in range(n-1, -1, -1):
        c[j] = z[j]-mu[j]*c[j+1]
        b[j] = (a[j+1]-a[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    yret = np.array([])
    for i in range(n):
        x = np.linspace(xs[i], xs[i+1], n_subdivide+1)[0:n_subdivide]
        f = a[i] + b[i]*(x-xs[i]) + c[i]*(x-xs[i])**2 + d[i]*(x-xs[i])**3
        yret = np.concatenate((yret, f))
    return yret

def poly_fit(X, xs, do_plot = False):
    """
    Given a Nx2 array X of 2D coordinates, fit an N^th order polynomial
    and evaluate it at the coordinates in xs.
    This function assumes that all of the points have a unique X position
    """
    x = X[:, 0]
    y = X[:, 1]
    N = X.shape[0]
    A = np.zeros((N, N))
    for i in range(N):
        A[:, i] = x**i
    AInv = np.linalg.inv(A)
    b = AInv.dot(y[:, None])

    M = xs.size
    Y = np.zeros((M, 2))
    Y[:, 0] = xs
    for i in range(N):
        Y[:, 1] += b[i]*(xs**i)
    if do_plot:
        plt.plot(Y[:, 0], Y[:, 1], 'b')
        plt.hold(True)
        plt.scatter(X[:, 0], X[:, 1], 20, 'r')
        plt.show()
    return Y

def draw_curve(X, Y, linewidth, color='k'):
    """
    Draw a parabolic curve between two 2D points
    Parameters
    ----------
    X: list of [x, y]
        First point
    Y: list of [x, y]
        Second point
    linewidth: int
        Thickness of line
    color: string
        Color to draw
    """
    if Y[1] < X[1]:
        X, Y = Y, X
    [x1, y1, x3, y3] = [X[0], X[1], Y[0], Y[1]]
    x2 = 0.5*x1 + 0.5*x3
    y2 = 0.25*y1 + 0.75*y3
    xs = np.linspace(x1, x3, 50)
    X = np.array([[x1, y1], [x2, y2], [x3, y3]])
    Y = poly_fit(X, xs, do_plot=False)
    plt.plot(Y[:, 0], Y[:, 1], color, linewidth=linewidth)

if __name__ == '__main__':
    [x1, y1, x3, y3] = [100, 100, 101, 104]
    x2 = 0.5*(x1 + x3)
    y2 = 0.25*y1 + 0.75*y3
    xs = np.linspace(x1, x3, 50)
    X = np.array([[x1, y1], [x2, y2], [x3, y3]])
    Y = poly_fit(X, xs, do_plot=False)
    plt.plot(Y[:, 0], Y[:, 1], 'k')
    plt.scatter(X[:, 0], X[:, 1], 20)
    plt.axis('equal')
    plt.show()
