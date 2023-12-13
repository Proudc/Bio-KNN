import numpy as np

import torch
from torch.nn import functional

from scipy.spatial.distance import cdist

def euclidean_torch(x, y):
    """
    Computes the Euclidean distance between two tensors.
    """
    return torch.norm(x - y, dim = -1)

def euclidean_numpy(x, y):
    """
    Computes the Euclidean distance between two numpy arrays.
    """
    return np.linalg.norm(x - y)


def euclidean_torch_separate(x, y, channel = 8):
    N = len(x)
    x = x.view(N, -1, channel)
    y = y.view(N, -1, channel)
    return torch.norm(x - y, dim = -1)


def euclidean_torch_separate_sum(x, y, channel = 8):
    N = len(x)
    x = x.view(N, -1, channel)
    y = y.view(N, -1, channel)

    return torch.norm(x - y, dim = -1).sum(dim=1)




def cosine_torch(x, y):
    return 1 - torch.cosine_similarity(x, y)

def cosine_numpy(x, y):
    """
    Computes the cosine distance between two numpy arrays.
    """
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))



def manhattan_torch(x, y):
    return functional.pairwise_distance(x, y, p = 1)

def manhattan_torch_separate(x, y, channel = 8):
    N = len(x)
    x = x.view(N, -1, channel)
    y = y.view(N, -1, channel)
    return functional.pairwise_distance(x, y, p = 1)

def manhattan_torch_separate_sum(x, y, channel = 8):
    N = len(x)
    x = x.view(N, -1, channel)
    y = y.view(N, -1, channel)
    return functional.pairwise_distance(x, y, p = 1).sum(dim=1)



def hyperbolic_torch(u, v, epsilon=1e-7):  # changed from epsilon=1e-7 to reduce error
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = torch.sqrt(x ** 2 - 1)
    return torch.log(x + z)

def hyperbolic_torch_separate(x, y, channel = 8, epsilon=1e-7):
    N = len(x)
    x = x.view(N, -1, channel)
    y = y.view(N, -1, channel)
    return hyperbolic_torch(x, y)

def hyperbolic_torch_separate_sum(x, y, channel = 8, epsilon=1e-7):
    N = len(x)
    x = x.view(N, -1, channel)
    y = y.view(N, -1, channel)
    return hyperbolic_torch(x, y).sum(dim=1)


def hyperbolic_numpy(u, v, epsilon=1e-9):
    sqdist = np.sum((u - v) ** 2, axis=-1)
    squnorm = np.sum(u ** 2, axis=-1)
    sqvnorm = np.sum(v ** 2, axis=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = np.sqrt(x ** 2 - 1)
    return np.log(x + z)



def l2_dist(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == q.shape[1]

    x = x.T
    sqr_q = np.sum(q ** 2, axis=1, keepdims=True)
    sqr_x = np.sum(x ** 2, axis=0, keepdims=True)
    l2 = sqr_q + sqr_x - 2 * q @ x
    l2[ np.nonzero(l2 < 0) ] = 0.0
    return np.sqrt(l2)

def cosine_dist(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == x.shape[1]

    qq = np.sum(q * q, axis=1) ** 0.5
    xx = np.sum(x * x, axis=1) ** 0.5
    q = q / qq[:, np.newaxis]
    x = x / xx[:, np.newaxis]
    return 1 - np.dot(q, x.T)

def manhattan_dist(q: np.ndarray, x: np.ndarray):
    result = cdist(q, x, metric='cityblock')
    return result

def hyperbolic_dist(u, v, epsilon=1e-9):
    assert len(u.shape) == 2
    assert len(v.shape) == 2
    assert u.shape[1] == v.shape[1]
    v = v.T
    sqr_u = np.sum(u ** 2, axis=1, keepdims=True)
    sqr_v = np.sum(v ** 2, axis=0, keepdims=True)
        
    sqdist = sqr_u + sqr_v - 2 * u @ v
    squnorm = np.sum(u ** 2, axis=-1, keepdims=True)
    sqvnorm = np.sum(v ** 2, axis=0, keepdims=True)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = np.sqrt(x ** 2 - 1)
    return np.log(x + z)

def l2_dist_separate(q: np.ndarray, x: np.ndarray, dataset_type):
    total_dis = np.zeros((len(q), len(x)))
    if dataset_type == "big_uniref":
        length = 24
    elif dataset_type == "big_uniprot":
        length = 25
    elif dataset_type == "protein":
        length = 20
    elif dataset_type == "dna":
        length = 4
    elif dataset_type == "big_qiita":
        length = 4
    elif dataset_type == "big_rt988":
        length = 4
    elif dataset_type == "big_gen50ks":
        length = 4
    else:
        raise ValueError("dataset_type error")
    tem_length = int(q.shape[1] / length)
    for i in range(length):
        tem_q = q[:, i * tem_length : (i + 1) * tem_length]
        tem_x = x[:, i * tem_length : (i + 1) * tem_length]
        total_dis += l2_dist(tem_q, tem_x)
    return total_dis

def manhattan_dist_separate(q: np.ndarray, x: np.ndarray, dataset_type):
    total_dis = np.zeros((len(q), len(x)))
    if dataset_type == "big_uniref":
        length = 24
    elif dataset_type == "big_uniprot":
        length = 25
    elif dataset_type == "protein":
        length = 20
    elif dataset_type == "dna":
        length = 4
    elif dataset_type == "big_qiita":
        length = 4
    elif dataset_type == "big_rt988":
        length = 4
    elif dataset_type == "big_gen50ks":
        length = 4
    else:
        raise ValueError("dataset_type error")
    tem_length = int(q.shape[1] / length)
    for i in range(length):
        total_dis += manhattan_dist(q[:, i * tem_length : (i + 1) * tem_length], x[:, i * tem_length : (i + 1) * tem_length])
    return total_dis


def hyperbolic_dist_separate(q: np.ndarray, x: np.ndarray, dataset_type):
    total_dis = np.zeros((len(q), len(x)))
    if dataset_type == "big_uniref":
        length = 24
    elif dataset_type == "big_uniprot":
        length = 25
    elif dataset_type == "protein":
        length = 20
    elif dataset_type == "dna":
        length = 4
    elif dataset_type == "big_qiita":
        length = 4
    elif dataset_type == "big_rt988":
        length = 4
    elif dataset_type == "big_gen50ks":
        length = 4
    else:
        raise ValueError("dataset_type error")
    tem_length = int(q.shape[1] / length)
    for i in range(length):
        total_dis += hyperbolic_dist(q[:, i * tem_length : (i + 1) * tem_length], x[:, i * tem_length : (i + 1) * tem_length])
    return total_dis


