import time
import numpy as np
import argparse
from mergetree import *
from matching import *
from matching_plots import *
from dtw import *
from utils import *
from evaluation import *
from curvature import *
import time
import gudhi # For bottleneck
import scipy.io as sio
import glob

def euclidean_shift_compare(x, y):
    N = max(len(x), len(y))
    # Do uniform scaling to make these the same length
    if len(x) < N:
        x = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(x)), x)
    if len(y) < N:
        y = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(y)), y)
    xf = np.fft.fft(x)
    yf = np.fft.fft(y)
    xy = np.max(np.real(np.fft.ifft(np.conj(yf)*xf)))
    return np.sum(x**2) + np.sum(y**2) - 2*xy



def get_distances(methods, dataset_path, results_path, batch_size, batch_index, sigma=10):
    """
    Compute all pairwise distances between the union of training and test data
    for a particular dataset

    Parameters
    ----------
    methods: dict {string-> fn: X, Y -> float}
        Dictionary of functions to compare time series
    dataset_path: string
        Path to mpeg7 dataset
    results_path: string
        Prefix to path to which to save results
    batch_size: int
        Number of rows for each batch
    batch_index: int
        Index of batch
    sigma: float
        Smoothing parameter
    """
    ## Step 1: Load in data
    filenames = glob.glob("{}/*.png".format(dataset_path))
    classes = set([])
    for f in filenames:
        f = f.split("mpeg7/")[-1]
        f = f.split("-")[0]
        classes.add(f)
    classes = sorted(list(classes))
    sigma = 10
    data = []
    for c in classes:
        for filename in sorted(glob.glob("{}/{}*.png".format(dataset_path, c))):
            X = get_contour_curve(filename)
            curv = get_curv_2d(X, sigma=sigma, loop=True)
            MT = MergeTree()
            MT.init_from_timeseries(curv, circular=True)
            data.append((MT, curv))
    
    ## Step 2: Compute similarities for these rows for all methods
    N = len(data)
    Ds = {name:np.zeros((batch_size, N)) for name in methods.keys()}
    for i in range(batch_size):
        row = batch_index*batch_size + i
        for j in range(row+1, N):
            for name, fn in methods.items():
                print(name, row, j)
                Ds[name][i, j] = fn(data[row], data[j])
        sio.savemat("{}/mpeg7_{}.mat".format(results_path, batch_index), Ds)


if __name__ == '__main__':
    circular=False
    methods = {}
    methods["circular_dope"] = lambda X, Y: circular_dope_match(X[1], Y[1])[0]
    methods["bottleneck"] = lambda X, Y: gudhi.bottleneck_distance(X[0].PD, Y[0].PD)
    methods["wasserstein"] = lambda X, Y: wasserstein(X[0].PD, Y[0].PD)
    methods["circular_dtw"] = lambda X, Y: dtw_cyclic(get_csm(X[1][0::4], Y[1][0::4]))[1]
    methods["euclidean"] = lambda X, Y: euclidean_shift_compare(X[1], Y[1])

    parser = argparse.ArgumentParser(description="Evaluating UCR dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datasetpath", type=str, action="store", default="./mpeg7", help="Path to dataset")
    parser.add_argument("-r", "--resultspath", type=str, action="store", default="./results", help="Path to results")
    parser.add_argument("-b", "--batchsize", type=int, action="store", default=2)
    parser.add_argument("-i", '--index', type=int, action="store", help="Batch index")
    cmd_args = parser.parse_args()

    get_distances(methods, cmd_args.datasetpath, cmd_args.resultspath, cmd_args.batchsize, cmd_args.index)