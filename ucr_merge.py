import argparse
from mergetree import *
from matching import *
from matching_plots import *
from dtw import *
from utils import *
from evaluation import *
import pyts
import pyts.datasets
import gudhi # For bottleneck
from ucr import *

TO_SKIP = ["Crop", "ElectricDevices"] # These datasets are too large

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
    

    parser = argparse.ArgumentParser(description="Merging UCR dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path", type=str, action="store", default="./results", help="Path to results")
    parser.add_argument("-i", '--index', type=int, action="store", help="UCR dataset index")
    parser.add_argument("-n", '--n_batches', type=int, action="store", help="Number of batches", default=50)
    cmd_args = parser.parse_args()

    dataset_name = pyts.datasets.ucr_dataset_list()[cmd_args.index]
    dataset = get_dataset(dataset_name, all_eps, circular=False)

    if not dataset_name in TO_SKIP:
        print("Merging", dataset_name)
        merge_dataset_batches(dataset_name, dataset, methods, all_eps, cmd_args.n_batches, cmd_args.path)
