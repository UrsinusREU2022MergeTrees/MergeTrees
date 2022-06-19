"""
Implement simple polynomial interpolation to help draw smooth curves
on the merge trees
"""
import numpy as np
import matplotlib.pyplot as plt


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

def draw_curve(X, Y, linewidth):
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
    """
    if Y[1] < X[1]:
        X, Y = Y, X
    [x1, y1, x3, y3] = [X[0], X[1], Y[0], Y[1]]
    x2 = 0.5*x1 + 0.5*x3
    y2 = 0.25*y1 + 0.75*y3
    xs = np.linspace(x1, x3, 50)
    X = np.array([[x1, y1], [x2, y2], [x3, y3]])
    Y = poly_fit(X, xs, do_plot=False)
    plt.plot(Y[:, 0], Y[:, 1], 'k', linewidth=linewidth)

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
