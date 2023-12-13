import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

def get_bin_idx(label):
    return int(label / 0.02)


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))
    return kernel_window

if __name__ == "__main__":
    needle_similarity_matrix = pickle.load(open("/home/zju/czh/clustering/protein/code/python/Neuprotein/5000_needle_512/similarity_matrix_result", "rb"))
    labels = np.concatenate(needle_similarity_matrix / 100)
    # labels = np.concatenate(needle_similarity_matrix / 100)
    
    bin_index_per_label = [get_bin_idx(label) for label in labels]
    
    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1
    # print("Nb: ", Nb)
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    # print("num_samples_of_bins: ", num_samples_of_bins)
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    # print("emp_label_dist: ", emp_label_dist)
    
    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
    print(max(eff_label_dist))
    weights = [np.float32(1 / x) for x in eff_label_dist]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]