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

def get_mean_reciprocal_rank(D, idx_train, idx_test, do_min=True):
    """
    Compute the mean reciprocal rank in the training set for every test item

    Parameters
    ----------
    D: ndarray(M, N)
        Distances to examine.  Training along rows and testing along columns
    idx_train: ndarray(M)
        Class indices of the training set
    idx_test: ndarray(N)
        Class indices of the test set
    do_min: boolean
        If true, report the MRR for the top ranked item.  If False, report
        MRR for all items
    """
    idx = np.argsort(D, axis=0)
    results = idx_train[idx] == idx_test[None, :]
    results = np.array(results, dtype=float)
    results[results == 0] = np.nan
    results = results*(np.arange(results.shape[0])[:, None])
    ret = 0
    if do_min:
        ret = np.mean(1/np.nanmin(1+results, axis=0))
    else:
        ret = np.mean(np.nanmean(1/(1+results), axis=0))
    return ret



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
    statistics = [  ("MR", get_mean_rank), 
                    ("MRR", get_mean_reciprocal_rank), 
                    ("MRR_ALL", lambda D, idx_train, idx_test: get_mean_reciprocal_rank(D, idx_train, idx_test, do_min=False)), 
                    ("MAP", get_map), ("1NN", get_1nn_error_rate)
                ]
    methods = ['dope', 'bottleneck', 'wasserstein', 'euclidean', 'dtw_full']

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


def evaluate_ucr_map_all():
    """
    Run all evaluation statistics across all methods across all datasets in the UCR dataset
    """
    import glob
    from ucr import get_dataset, unpack_D, get_firstlast_dist
    import scipy.io as sio
    # Report mean rank of training data for every test item
    datasets = [f.split("_dtw")[0].split("/")[-1] for f in glob.glob("results/*dtw*")]
    methods = ['dope', 'bottleneck', 'wasserstein', 'euclidean', 'dtw_full']
    fout = open("MAP_ALL.csv", "w")
    fout.write("Dataset,")
    for i, method in enumerate(methods):
        fout.write(method)
        if i < len(methods)-1:
            fout.write(",")
        else:
            fout.write("\n")
    for dataset_str in datasets:
        print(dataset_str)
        fout.write("{},".format(dataset_str))
        dataset = get_dataset(dataset_str, [])
        D_firstlast = get_firstlast_dist(dataset)
        idx_train = dataset['target_train']
        idx_test = dataset['target_test']
        idx_all = np.concatenate((idx_train, idx_test))
        for i, method in enumerate(methods):
            ## Step 1: Do the default parameter for this method
            dataset_method = {**dataset, **sio.loadmat("results/{}_{}_0.0.mat".format(dataset_str, method))}
            D = unpack_D(dataset_method['D'])
            if (not "dtw" in method) and (not "euclidean" in method):
                D += D_firstlast
            res = get_map(D, idx_all, idx_all)
            fout.write("{}".format(res))
            if i < len(methods)-1:
                fout.write(",")
            else:
                fout.write("\n")
        fout.flush()
    fout.close()

def make_critical_distance_plot(data_path, alpha):
    """
    Make a critical distance plot for a set of classifiers using
    pairwise Wilcoxon signed tests with a Holm-Bonferroni correction

    Parameters
    ----------
    data_path: string
        Path to csv file containing results
    alpha: float
        Cutoff p-value to use
    """
    import pandas as pd
    from scipy.stats import wilcoxon
    import matplotlib.pyplot as plt

    data = pd.read_csv(data_path)
    methods = ['dope', 'wasserstein', 'bottleneck', 'euclidean', 'dtw_full']
    colors = {m:"C%i"%i for i, m in enumerate(methods)}
    N = len(methods)
    methods_data = [data[m].to_numpy() for m in methods]
    avgs = np.array([np.mean(data[m].to_numpy()) for m in methods])
    methods = [methods[i] for i in np.argsort(avgs)]
    methods_data = [methods_data[i] for i in np.argsort(avgs)]
    avgs = np.sort(avgs)

    ## Step 1: Plot each classifier
    plt.scatter(avgs, 0*avgs, c='k')
    xticks = plt.gca().get_xticks()
    dx = xticks[-1]-xticks[0]
    plt.plot([xticks[0], xticks[-1]], [0, 0], c='k')
    dy = 0.05*dx
    for i, avg in enumerate(avgs):
        h = i # Height towards bottom
        if i > N // 2:
            h = N-i
        h += 1
        y = -h*dy
        x = xticks[0]-0.3*dx
        if i > N // 2:
            x = xticks[-1]+0.2*dx
        color = colors[methods[i]]
        plt.plot([avg, avg], [0, y], color)
        plt.plot([x, avg], [y, y], color)
        plt.scatter([avg], [0], c=color, zorder=100)
        if i > N // 2:
            x = avgs[-1] + 0.1*(xticks[1]-xticks[0])
        plt.text(x, y+0.2*dy, "{} ({:.4f})".format(methods[i], avg), c=color)
    
    for tick in xticks:
        plt.plot([tick, tick], [-dy/8, dy/8], c='k')
        plt.text(tick-0.2*(xticks[1]-xticks[0]), dy/5, "{:.2f}".format(tick))
    
    ## Step 2: Plot merges for things that are statistically similar
    ps = []
    for i in range(N):
        for j in range(i+1, N):
            p = wilcoxon(methods_data[i], methods_data[j], zero_method="zsplit")[-1]
            ps.append([methods[i], methods[j], p])
    ps = sorted(ps, key=lambda x: x[-1])
    for i, x in enumerate(ps):
        fac = alpha/(len(ps)-i) # Holm-Bonferroni correction
        print(x, x[-1] < fac)
    
    plt.axis('off')

if __name__ == '__main__':
    evaluate_ucr()