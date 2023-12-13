import numpy as np
import pickle
from scipy.stats import wasserstein_distance

import random

from sklearn.cluster import DBSCAN as skDBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans as skKmeans
from sklearn.cluster import SpectralClustering as skSpectral
import matplotlib.pyplot as plt
import os
import sys

from shapely.geometry import Polygon, MultiPolygon


def invert(origin_img):
    result_img = np.zeros((origin_img.shape[1], origin_img.shape[0]))
    for i in range(origin_img.shape[0]):
        result_img[:, i] = origin_img[i]
    return result_img

def agglomerative(dist_matrix, cell_size, train_seq_num, root_path, n_clusters = None, distance_threshold = None):
    cell_num = int(train_seq_num / cell_size)
    if n_clusters != None:
        cluster_result = AgglomerativeClustering(n_clusters = n_clusters, linkage='complete', affinity = "precomputed").fit(dist_matrix)
    elif distance_threshold != None:
        cluster_result = AgglomerativeClustering(distance_threshold = distance_threshold, linkage='complete', affinity = "precomputed").fit(dist_matrix)
    else:
        raise ValueError("Error")
    
    write_path = root_path + "agglomerative_" + str(train_seq_num) + "_" + str(cell_size) + "_" + str(n_clusters) + "_" + str(distance_threshold) + ".jpg"
    t = convert_list_to_matrix(cluster_result.labels_, cell_num, cell_num)
    plt.imshow(invert(t), origin = "lower", cmap = "GnBu")
    # plt.colorbar()
    plt.savefig(write_path, dpi=300)
    return cluster_result.labels_

def kmeans(dist_matrix, kmeans_k, cell_size, train_seq_num, root_path):
    cell_num = int(train_seq_num / cell_size)
    cluster_result = skKmeans(n_clusters = kmeans_k, init="random", random_state = random.randint(0, 200)).fit(dist_matrix)
    
    write_path = root_path + "kmeans_" + str(train_seq_num) + "_" + str(cell_size) + "_" + str(kmeans_k) + ".jpg"
    t = convert_list_to_matrix(cluster_result.labels_, cell_num, cell_num)
    plt.imshow(invert(t), origin = "lower", cmap = "GnBu")
    # plt.colorbar()
    plt.savefig(write_path, dpi=300)
    return cluster_result.labels_


def dbscan(dist_matrix, dbscan_eps, dbscan_min_samples, cell_size, train_seq_num, root_path):
    cell_num = int(train_seq_num / cell_size)
    cluster_result = skDBSCAN(eps = dbscan_eps, min_samples = dbscan_min_samples, metric = "precomputed").fit(dist_matrix)
    
    write_path = root_path + "dbscan_" + str(train_seq_num) + "_" + str(cell_size) + "_" + str(dbscan_eps) + "_" + str(dbscan_min_samples) + ".jpg"
    t = convert_list_to_matrix(cluster_result.labels_, cell_num, cell_num)
    plt.imshow(invert(t), origin = "lower", cmap = "GnBu")
    # plt.colorbar()
    plt.savefig(write_path, dpi=300)
    return cluster_result.labels_

def spectral(dist_matrix, cluster_num, cell_size, train_seq_num, root_path):
    cell_num = int(train_seq_num / cell_size)
    cluster_result = skSpectral(n_clusters = cluster_num, affinity = "precomputed", random_state=random.randint(0, 200)).fit(1 - dist_matrix)
    write_path = root_path + "spectral_" + str(train_seq_num) + "_" + str(cell_size) + "_" + str(cluster_num) + ".jpg"
    t = convert_list_to_matrix(cluster_result.labels_, cell_num, cell_num)
    plt.imshow(invert(t), origin = "lower", cmap = "GnBu")
    # plt.colorbar()
    plt.savefig(write_path, dpi=300)
    return cluster_result.labels_





def convert_matrix_to_list(input_matrix):
    output_list = []
    for i in range(len(input_matrix)):
        for j in range(len(input_matrix[i])):
            output_list.append(input_matrix[i][j])
    return output_list

def cal_dist_matrix(input_list):
    dist_matrix = np.zeros((len(input_list), len(input_list)))
    for i, value1 in enumerate(input_list):
        for j, value2 in enumerate(input_list):
            dis1 = wasserstein_distance([k / 1000 for k in range(1001)], [k / 1000 for k in range(1001)], value1[0], value2[0])
            dis2 = wasserstein_distance([k / 1000 for k in range(1001)], [k / 1000 for k in range(1001)], value1[1], value2[1])
            dist_matrix[i][j] = dis1 + dis2
    return dist_matrix

def cal_coor_to_origin(input):
    coor = []
    for i, value in enumerate(input):
        dis1 = wasserstein_distance([k / 1000 for k in range(1001)], [k / 1000 for k in range(1001)], value[0], [k / 1000 for k in range(1001)])
        dis2 = wasserstein_distance([k / 1000 for k in range(1001)], [k / 1000 for k in range(1001)], value[1], [k / 1000 for k in range(1001)])
        coor.append([dis1, dis2])
    return coor

def convert_list_to_matrix(input, sizex, sizey):
    matrix = np.zeros((sizex, sizey)) - 1
    pos = 0
    for i in range(sizex):
        for j in range(i, sizex):
            matrix[i][j] = input[pos]
            pos += 1
    return matrix


def count_cell(pb, pe, nb, ne, distance_matrix, train_seq_num):
    p_dist_list = []
    n_dist_list = []
    for i in range(train_seq_num):
        s_list = distance_matrix[i]
        for p in range(pb, pe):
            for n in range(nb, ne):
                if p >= n:
                    continue
                p_dist_list.append(s_list[p])
                n_dist_list.append(s_list[n])
    return p_dist_list, n_dist_list


def merge_cluster(label_result, coor_list):
    label_result = list(label_result)
    result_list = []
    tem_result = []
    tem_label_list = []
    for i, coor in enumerate(coor_list):
        tem_label = label_result[i]
        if tem_label in tem_label_list:
            index = tem_label_list.index(tem_label)
            flag = 0
            for ploy in tem_result[index]:
                merged_geometry = ploy.union(Polygon(coor))
                if isinstance(merged_geometry, Polygon):
                    flag = 1
                    break
            if flag == 0:
                tem_result[index].append(Polygon(coor))
            else:
                tem_result[index].append(merged_geometry)
        else:
            tem_result.append([Polygon(coor)])
            tem_label_list.append(tem_label)
    for i in range(len(tem_result)):
        curr_list = tem_result[i]
        if len(curr_list) == 1:
            result_list.append(curr_list[0].exterior.coords)
        else:
            first_result = curr_list[0].union(curr_list[1])
            for j in range(2, len(curr_list)):
                first_result = first_result.union(curr_list[j])
                
            result_list.append(first_result.exterior.coords)
    return result_list


def get_rec(cell_size, train_seq_num):
    length = int(train_seq_num / cell_size)
    rec_matrix = []
    for i in range(length):
        pb = int(cell_size * i)
        pe = int(cell_size * (i + 1))
        column_list = []
        for j in range(i, length):
            nb = int(cell_size * j)
            ne = int(cell_size * (j + 1))
            coor = [(pb, nb), (pe, nb), (pe, ne), (pb, ne)]
            column_list.append(coor)

        rec_matrix.append(column_list)
    return rec_matrix
    



def get_2d_coor(cell_size, train_seq_num, distance_matrix):
    sort_distance_matrix = np.zeros((train_seq_num, train_seq_num))
    for i in range(train_seq_num):
        distance_list = distance_matrix[i]
        o_list = [[j, distance_list[j]] for j in range(train_seq_num)]
        s_list = sorted(o_list, key = lambda a: a[1], reverse = False)
        value = [t[1] for t in s_list]
        sort_distance_matrix[i] = np.array(value)
    length = int(train_seq_num / cell_size)
    distribution_matrix = []
    for i in range(length):
        pb = int(cell_size * i)
        pe = int(cell_size * (i + 1))
        column_list = []
        for j in range(i, length):
            nb = int(cell_size * j)
            ne = int(cell_size * (j + 1))
            p_dist, n_dist = count_cell(pb, pe, nb, ne, sort_distance_matrix, train_seq_num)
            p_tem = [0 for k in range(1001)]
            n_tem = [0 for k in range(1001)]
            if len(p_dist) == 0 and len(n_dist) == 0:
                column_list.append([p_tem, n_tem])
            else:
                for p_value, n_value in zip(p_dist, n_dist):
                    p_tem[int(p_value * 1000)] += 1
                    n_tem[int(n_value * 1000)] += 1
                print(int(train_seq_num / cell_size) * int(train_seq_num / cell_size), i, j, np.sum(p_tem), np.sum(n_tem))

                p_tem = [value / np.sum(p_tem) for value in p_tem]
                n_tem = [value / np.sum(n_tem) for value in n_tem]
                column_list.append([p_tem, n_tem])
        print(len(column_list))
        distribution_matrix.append(column_list)
    return distribution_matrix

def grid_split_4(root_path):
    area1 = [[0, 0], [500, 500], [0, 500]]
    area2 = [[0, 500], [500, 500], [0, 1000]]
    area3 = [[0, 1000], [500, 500], [500, 1000]]
    area4 = [[500, 1000], [1000, 1000], [500, 500]]
    total_result = [area1, area2, area3, area4]
    write_path = root_path + "grid_4"
    print(write_path)
    pickle.dump(total_result, open(write_path, "wb"))

def grid_split_2(root_path):
    area1 = [[0, 0], [500, 500], [0, 1000]]
    area2 = [[500, 500], [1000, 1000], [0, 1000]]
    total_result = [area1, area2]
    write_path = root_path + "grid_2"
    print(write_path)
    pickle.dump(total_result, open(write_path, "wb"))


if __name__ == "__main__":
    dataset     = sys.argv[1] 
    cluster_num = sys.argv[2]
    root_path = "./"
    cell_size = 100
    distance_matrix = 1 - pickle.load(open(root_path + dataset + "/train_similarity_matrix_result", "rb")) / 100
    
    train_seq_num = 1000
    if not os.path.exists(root_path + dataset + "/clustering/"):
        os.mkdir(root_path + dataset + "/clustering/")

    if os.path.exists(root_path + dataset + "/clustering/distribution_matrix_" + str(cell_size) + "_" + str(train_seq_num)):
        print("Distribution matrix exist, begin load...")
        distribution_matrix = pickle.load(open(root_path + dataset + "/clustering/distribution_matrix_" + str(cell_size) + "_" + str(train_seq_num), "rb"))
    else:  
        print("Distribution matrix not exist, begin cal...")
        distribution_matrix = get_2d_coor(cell_size, train_seq_num, distance_matrix)
        pickle.dump(distribution_matrix, open(root_path + dataset + "/clustering/distribution_matrix_" + str(cell_size) + "_" + str(train_seq_num), "wb"))
    
    distribution_list = convert_matrix_to_list(distribution_matrix)
    coor_list = convert_matrix_to_list(get_rec(cell_size, train_seq_num))

    if os.path.exists(root_path + dataset + "/clustering/dist_matrix_" + str(cell_size) + "_" + str(train_seq_num)):
        print("Dist matrix exist, begin load...")
        dist_matrix = pickle.load(open(root_path + dataset + "/clustering/dist_matrix_" + str(cell_size) + "_" + str(train_seq_num), "rb"))
    else:  
        print("Dist matrix not exist, begin cal...")
        dist_matrix = cal_dist_matrix(distribution_list)
        pickle.dump(dist_matrix, open(root_path + dataset + "/clustering/dist_matrix_" + str(cell_size) + "_" + str(train_seq_num), "wb"))

    root_path = "./" + str(dataset) + "/clustering/"
    print("Begin agg clustering...")
    n = int(cluster_num)
    thres = None
    labels = agglomerative(dist_matrix, cell_size, train_seq_num, root_path, n_clusters = n, distance_threshold = thres)
    merge_result = merge_cluster(labels, coor_list)
    write_path = root_path + "agglomerative_" + str(train_seq_num) + "_" + str(cell_size) + "_" + str(n) + "_" + str(thres)
    print(write_path)
    pickle.dump(merge_result, open(write_path, "wb"))


    print("Begin spectral clustering...")
    cluster_num = int(cluster_num)
    labels = spectral(dist_matrix, cluster_num, cell_size, train_seq_num, root_path)
    merge_result = merge_cluster(labels, coor_list)
    write_path = root_path + "spectral_" + str(train_seq_num) + "_" + str(cell_size) + "_" + str(cluster_num)
    print(write_path)
    pickle.dump(merge_result, open(write_path, "wb"))

    print("Begin kmeans clustering...")
    kmeans_k = int(cluster_num)
    labels = kmeans(dist_matrix, kmeans_k, cell_size, train_seq_num, root_path)
    merge_result = merge_cluster(labels, coor_list)
    write_path = root_path + "kmeans_" + str(train_seq_num) + "_" + str(cell_size) + "_" + str(kmeans_k)
    print(write_path)
    pickle.dump(merge_result, open(write_path, "wb"))

    grid_split_4(root_path)
    grid_split_2(root_path)