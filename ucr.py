import time
import numpy as np
import argparse
from mergetree import *
from matching import *
from matching_plots import *
from dtw import *
from utils import *
from evaluation import *
import pyts
import pyts.datasets
import time
import gudhi # For bottleneck
import scipy.io as sio
import os
import glob
from sys import argv

TO_SKIP = ["Crop", "ElectricDevices"] # These datasets are too large

def euclidean_compare(x, y):
    N = max(len(x), len(y))
    # Zeropadding, as suggested by Eammon
    if len(x) < N:
        x = np.concatenate((x, np.zeros(N-len(x))))
    if len(y) < N:
        y = np.concatenate((y, np.zeros(N-len(y))))
    return np.sum((x-y)**2)

def get_dataset(dataset_name, all_eps, circular=False):
    """
    Load in a UCR dataset and create all merge trees and simplifications

    Parameters
    ----------
    dataset_name: string
        Name of UCR dataset
    all_eps: ndarray(K)
        List of all epsilon simplifications to make
    circular: boolean
        Whether to use circular boundary conditions for sublevelset filtrations
    
    Returns
    -------
    dataset: dict
        UCR dataset
    """
    dataset = pyts.datasets.fetch_ucr_dataset(dataset_name)
    for s in ["train", "test"]:
        idx = np.argsort(dataset['target_'+s])
        dataset['target_'+s] = dataset['target_'+s][idx]
        #dataset['data_'+s] = [znormalize(x) for x in dataset['data_'+s][idx]]
        dataset['data_'+s] = [x[~np.isnan(x)] for x in dataset['data_'+s][idx]]
        for i, x in enumerate(dataset['data_'+s]):
            MT = MergeTree()
            MT.init_from_timeseries(x, circular=circular)
            dataset['data_'+s][i] = (MT, x)
        for eps in all_eps:
            skey = 'data_{}_{:.1f}'.format(s, eps)
            dataset[skey] = []
            for x in dataset['data_'+s]:
                MT = MergeTree()
                MT.init_from_timeseries(x[1], circular=circular)
                MT.persistence_simplify(eps)
                x = x[1]
                if eps > 0:
                    x = MT.get_rep_timeseries()['ys']
                dataset[skey].append((MT, x))
    return dataset

def get_dataset_distances(dataset_name, dataset, methods, all_eps, batch_idx, n_batches, prefix="."):
    """
    Compute all pairwise distances between the union of training and test data
    for a particular dataset

    Parameters
    ----------
    dataset_name: string
        Name of the dataset
    dataset: dictionary
        Information with the dataset
    methods: dictionary {string: function handle}
        Methods to compute distances
    all_eps: list of float
        All epsilons to try
    batch_idx: int
        Index of batch to perform
    n_batches: int
        Total number of batches into which to split each job
    prefix: string
        Path to results file
    """
    M = len(dataset['data_train'])
    N = len(dataset['data_test'])
    print(dataset.keys())
    for method_name, method in methods.items():
        for eps in all_eps:
            if eps > 0 and ("dtw" in method_name or "euclidean" in method_name):
                continue
            eps = "{:.1f}".format(eps)
            XTrain = dataset['data_train_{}'.format(eps)]
            XTest = dataset['data_test_{}'.format(eps)]            
            tic = time.time()
            filename = "{}/{}_{}_{}.mat".format(prefix, dataset_name, method_name, eps)
            if os.path.exists(filename):
                print("Skipping", filename)
                continue
            filename = "{}/{}_{}_{}_{}.mat".format(prefix, dataset_name, method_name, batch_idx, eps)
            if os.path.exists(filename):
                print("Skipping", filename)
            else:
                print("\nTraining ", method_name, "on", dataset_name, ", eps = ", eps, ", batch ", batch_idx+1, "of", n_batches)
                batch_size = int(np.ceil(M/n_batches))
                print("batch_size", batch_size)
                D = np.zeros((batch_size, N))
                for i in range(batch_size):
                    if i%10 == 0 and i != 0:
                        print(".", end="", flush=True)
                    idx = batch_idx*batch_size+i
                    if idx < len(XTrain):
                        xi = XTrain[idx]
                        for j in range(N):
                            yj = XTest[j]
                            D[i, j] = method(xi, yj)
                    else:
                        D = D[0:i, :]
                sio.savemat(filename, {"D":D})
                print("Elapsed Time: {:.3f}".format(time.time()-tic))

def merge_dataset_batches(dataset_name, dataset, methods, all_eps, n_batches, prefix="."):
    """
    Compute all pairwise distances between the union of training and test data
    for a particular dataset

    Parameters
    ----------
    dataset_name: string
        Name of the dataset
    dataset: dictionary
        Information with the dataset
    methods: dictionary {string: function handle}
        Methods to compute distances
    all_eps: list of float
        All epsilons to try
    n_batches: int
        Total number of batches into which to split each job
    prefix: string
        Path to results file
    """
    M = len(dataset['data_train'])
    N = len(dataset['data_test'])
    batch_size = int(np.ceil(M/n_batches))
    for method_name in methods.keys():
        for eps in all_eps:
            if eps > 0 and ("dtw" in method_name or "euclidean" in method_name):
                continue
            eps = "{:.1f}".format(eps)
            filename = "{}/{}_{}_{}.mat".format(prefix, dataset_name, method_name, eps)
            if os.path.exists(filename):
                print("Skipping", filename)
                continue
            else:
                # If this is the last batch, merge them all together and remove sub-batches
                D = np.zeros((M, N))
                if len(glob.glob("{}/{}_{}_*_{}.mat".format(prefix, dataset_name, method_name, eps))) != n_batches:
                    print("Not enough batches found when attempting to merge", dataset_name)
                else:
                    print("Merging", filename)
                    for batch_idx in range(n_batches):
                        filename = "{}/{}_{}_{}_{}.mat".format(prefix, dataset_name, method_name, batch_idx, eps)
                        D[batch_idx*batch_size:(batch_idx+1)*batch_size, :] = sio.loadmat(filename)["D"]
                        os.remove(filename)
                    filename = "{}/{}_{}_{}.mat".format(prefix, dataset_name, method_name, eps)
                    sio.savemat(filename, {"D":D})

def get_firstlast_dist(dataset):
    """
    Return a distance matrix which is the sum of the absolute differences
    between the first and last points in a dataset
    """
    X = []
    for x in dataset['data_train'] + dataset['data_test']:
        x = x[1]
        X.append([x[0], x[-1]])
    X = np.array(X)
    left = X[:, 0]
    right = X[:, 1]
    D = np.abs(left[:, None] - left[None, :]) + np.abs(right[:, None] - right[None, :])
    return D

def unpack_D(d):
    """
    Unpack the lower triangular compressed distance matrix from the
    get_dataset_distances method
    """
    N = 1+(1+8*d.size)**0.5
    N = int(N/2)
    D = np.zeros((N, N))
    pix = np.arange(N)
    I, J = np.meshgrid(pix, pix, indexing='ij')
    D[I > J] = d.flatten()
    D = D + D.T
    return D


if __name__ == '__main__':
    circular=False
    all_eps = [0]
    methods = {}
    methods["dope"] = lambda X, Y: dope_match(X[1], Y[1], circular=circular)[0]
    methods["bottleneck"] = lambda X, Y: gudhi.bottleneck_distance(X[0].PD, Y[0].PD)
    methods["wasserstein"] = lambda X, Y: wasserstein(X[0].PD, Y[0].PD)
    methods["dtw_full"] = lambda X, Y: cdtw(X[1], Y[1], compute_path=False)[0]
    methods["dtw_crit"] = lambda X, Y: cdtw(X[1], Y[1], compute_path=False)[0]
    methods["euclidean"] = lambda X, Y: euclidean_compare(X[0].crit, Y[0].crit)
    

    parser = argparse.ArgumentParser(description="Evaluating UCR dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path", type=str, action="store", default="./results", help="Path to results")
    parser.add_argument("-i", '--index', type=int, action="store", help="UCR dataset index")
    parser.add_argument("-o", '--offset', type=int, action="store", help="Batch Offset", default=0)
    parser.add_argument("-n", '--n_batches', type=int, action="store", help="Number of batches", default=50)
    cmd_args = parser.parse_args()

    idx = cmd_args.index + cmd_args.offset
    dataset_idx = idx // cmd_args.n_batches
    batch_idx = idx % cmd_args.n_batches

    dataset_name = pyts.datasets.ucr_dataset_list()[dataset_idx]
    print(dataset_name, batch_idx)
    dataset = get_dataset(dataset_name, all_eps, circular=False)

    if dataset_name in TO_SKIP:
        print("Skipping", dataset_name)
    else:
        print("Doing", dataset_name)
        get_dataset_distances(dataset_name, dataset, methods, all_eps, batch_idx, cmd_args.n_batches, cmd_args.path)
