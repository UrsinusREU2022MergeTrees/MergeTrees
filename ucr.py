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
from sys import argv

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

def get_dataset_distances(dataset_name, dataset, methods, prefix="."):
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
    prefix: string
        Path to results file
    """
    M = len(dataset['data_train'])
    N = len(dataset['data_test'])
    for method_name, method in methods.items():
        for eps in all_eps:
            if eps > 0 and ("dtw" in method_name or "euclidean" in method_name):
                continue
            XAll = dataset['data_train_{}'.format(eps)] + dataset['data_test_{}'.format(eps)]
            eps = "{:.1f}".format(eps)
            tic = time.time()
            D = np.zeros((M+N, M+N))
            filename = "{}/{}_{}_{}.mat".format(prefix, dataset_name, method_name, eps)
            if os.path.exists(filename):
                print("Skipping", filename)
            else:
                print("\nTraining ", method_name, "on", dataset_name, ", eps = ", eps)
                for i in range(M+N):
                    if i%10 == 0 and i != 0:
                        print(".", end="", flush=True)
                    xi = XAll[i]
                    for j in range(i+1, M+N):
                        yj = XAll[j]
                        D[i, j] = method(xi, yj)
                D = D + D.T
                pix = np.arange(M+N)
                I, J = np.meshgrid(pix, pix, indexing='ij')
                D = D[I > J]
                sio.savemat(filename, {"D":D})
                print("Elapsed Time: {:.3f}".format(time.time()-tic))

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
    all_eps = np.arange(0, 11)/10
    methods = {}
    methods["dope"] = lambda X, Y: dope_match(X[1], Y[1], circular=circular)[0]
    methods["bottleneck"] = lambda X, Y: gudhi.bottleneck_distance(X[0].PD, Y[0].PD)
    methods["wasserstein"] = lambda X, Y: wasserstein(X[0].PD, Y[0].PD)
    methods["dtw_full"] = lambda X, Y: cdtw(X[1], Y[1], compute_path=False)[0]
    methods["euclidean"] = lambda X, Y: euclidean_compare(X[1], Y[1])
    

    parser = argparse.ArgumentParser(description="Evaluating UCR dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path", type=str, action="store", default="./results", help="Path to results")
    parser.add_argument("-i", '--index', type=int, action="store", help="UCR dataset index")
    cmd_args = parser.parse_args()

    dataset_name = pyts.datasets.ucr_dataset_list()[cmd_args.index]
    dataset = get_dataset(dataset_name, all_eps, circular=False)
    print("Doing", dataset_name)
    get_dataset_distances(dataset_name, dataset, methods, cmd_args.path)
