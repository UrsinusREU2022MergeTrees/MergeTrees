import numpy as np
from numba import jit

def get_1nn_error_rate(D, idx_train, idx_test):
    """
    Compute the error rate of a 1 nearest neighbor classifier
    on a particular dataset

    Parameters
    ----------
    D: ndarray(M, N)
        Distances to examine.  Training along rows and testing along columns
    idx_train: ndarray(M)
        Class indices of the training set
    idx_test: ndarray(N)
        Class indices of the test set
    """
    correct = 0
    i = np.argmin(D, axis=0)
    correct = np.sum(idx_train[i] == idx_test)
    N = D.shape[1]
    return (N-correct)/N

def get_mean_rank(D, idx_train, idx_test):
    """
    Compute the mean rank of the first correctly identified item in the training
    set for every test item

    Parameters
    ----------
    D: ndarray(M, N)
        Distances to examine.  Training along rows and testing along columns
    idx_train: ndarray(M)
        Class indices of the training set
    idx_test: ndarray(N)
        Class indices of the test set
    """
    idx = np.argsort(D, axis=0)
    results = idx_train[idx] == idx_test[None, :]
    results = np.array(results, dtype=float)
    results[results == 0] = np.nan
    results = results*(np.arange(results.shape[0])[:, None])
    return np.mean(np.nanmin(1+results, axis=0))

def get_mean_reciprocal_rank(D, idx_train, idx_test):
    """
    Compute the mean reciprocal rank of the first correctly identified item
    in the training set for every test item

    Parameters
    ----------
    D: ndarray(M, N)
        Distances to examine.  Training along rows and testing along columns
    idx_train: ndarray(M)
        Class indices of the training set
    idx_test: ndarray(N)
        Class indices of the test set
    """
    idx = np.argsort(D, axis=0)
    results = idx_train[idx] == idx_test[None, :]
    results = np.array(results, dtype=float)
    results[results == 0] = np.nan
    results = results*(np.arange(results.shape[0])[:, None])
    return np.mean(1/np.nanmin(1+results, axis=0))



@jit(nopython=True)
def get_map_helper(idx, idx_train, idx_test):
    """
    Compute the mean average precision of every test item with respect to the
    training data

    Parameters
    ----------
    idx: ndarray(M, N)
        Result of running np.argsort(D, axis=0)
    idx_train: ndarray(M)
        Class indices of the training set. 
    idx_test: ndarray(N)
        Class indices of the test set. 
    """
    map = 0
    for j in range(idx.shape[1]):
        recall = 0
        ap = 0
        for i in range(idx.shape[0]):
            if idx_train[idx[i, j]] == idx_test[j]:
                ap += (recall+1) / (i+1)
                recall += 1
        map += ap/recall
    return map/idx.shape[1]


def get_map(D, idx_train, idx_test):
    """
    Compute the mean average precision of every test item with respect to the
    training data

    Parameters
    ----------
    D: ndarray(M, N)
        Distances to examine.  Training along rows and testing along columns
    idx_train: ndarray(M)
        Class indices of the training set. 
    idx_test: ndarray(N)
        Class indices of the test set. 
    """
    idx = np.argsort(D, axis=0)
    return get_map_helper(idx, idx_train, idx_test)

def evaluate_ucr():
    """
    Run all evaluation statistics across all methods across all datasets in the UCR dataset
    """
    import glob
    from ucr import get_dataset, unpack_D
    import scipy.io as sio
    # Report mean rank of training data for every test item
    datasets = [f.split("_dtw")[0].split("/")[-1] for f in glob.glob("results/*dtw*")]
    statistics = [("MR", get_mean_rank), ("MRR", get_mean_reciprocal_rank), ("MAP", get_map), ("1NN", get_1nn_error_rate)]
    methods = ['dope', 'bottleneck', 'wasserstein', 'dtw_full']

    for (stat_name, fn_stat) in statistics:
        fout = open("{}.csv".format(stat_name), "w")
        fout.write("Dataset,")
        for i, method in enumerate(methods):
            fout.write("{},{} (Best Param)".format(method, method))
            if i < len(methods)-1:
                fout.write(",")
            else:
                fout.write("\n")
        for dataset_str in datasets:
            print(stat_name, dataset_str)
            fout.write("{},".format(dataset_str))
            dataset = get_dataset(dataset_str, [])
            M = len(dataset['target_train'])
            idx_train = dataset['target_train']
            idx_test = dataset['target_test']
            for i, method in enumerate(methods):
                ## Step 1: Do the default parameter for this method
                dataset_method = {**dataset, **sio.loadmat("results/{}_{}_0.0.mat".format(dataset_str, method))}
                D = unpack_D(dataset_method['D'])
                res = fn_stat(D[0:M, M::], idx_train, idx_test)
                fout.write("{},".format(res))

                ## Step 2: Find the best parameter in the training data
                best_train_dataset = "results/{}_{}_0.0.mat".format(dataset_str, method)
                best_train_value = res
                for train_dataset in glob.glob("results/{}_{}_*.mat".format(dataset_str, method)):
                    dataset_method = {**dataset, **sio.loadmat("results/{}_{}_0.0.mat".format(dataset_str, method))}
                    D = unpack_D(dataset_method['D'])
                    res = fn_stat(D[0:M, 0:M], idx_train, idx_train)
                    if res < best_train_value:
                        best_train_value = res
                        best_train_dataset = train_dataset
                
                ## Step 3: Apply this parameter to the test data
                param = best_train_dataset.split("_")[-1][0:-4]
                dataset_method = {**dataset, **sio.loadmat(best_train_dataset)}
                D = unpack_D(dataset_method['D'])
                res = fn_stat(D[0:M, M::], idx_train, idx_test)
                fout.write("{} ({})".format(res, param))
                if i < len(methods)-1:
                    fout.write(",")
                else:
                    fout.write("\n")
            fout.flush()
        fout.close()

if __name__ == '__main__':
    evaluate_ucr()