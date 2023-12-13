import random
import numpy as np
from tools import function
import pickle


def main_triplet_selection(sampling_type, sampling_num, knn, distance_list, pred_distance_matrix, anchor_pos, anchor_length, length_list, epoch, my_config):
    if sampling_type == "distance_sampling":
        positive_sampling_index_list = distance_sampling(distance_list, sampling_num, "positive")
        negative_sampling_index_list = distance_sampling(distance_list, sampling_num, "negative")
    elif sampling_type == "distance_sampling2":
        positive_sampling_index_list, negative_sampling_index_list = distance_sampling2(knn, sampling_num)
    elif sampling_type == "distance_sampling3":
        positive_sampling_index_list, negative_sampling_index_list = distance_sampling3(knn, sampling_num, distance_list)
    elif sampling_type == "distance_sampling4":
        positive_sampling_index_list, negative_sampling_index_list = distance_sampling4(distance_list, anchor_pos, sampling_num, anchor_length, length_list)
    elif sampling_type == "distance_sampling5":
        positive_sampling_index_list, negative_sampling_index_list = distance_sampling5(distance_list, anchor_pos, sampling_num, anchor_length, length_list)
    elif sampling_type == "distance_sampling6":
        positive_sampling_index_list, negative_sampling_index_list = distance_sampling6(knn, sampling_num, distance_list)
    elif sampling_type == "distance_sampling7":
        positive_sampling_index_list, negative_sampling_index_list = distance_sampling7(knn, pred_distance_matrix[anchor_pos])
    elif sampling_type == "distance_sampling8":
        raise ValueError("8 is not supported!")
    elif sampling_type == "distance_sampling9":
        raise ValueError("9 is not supported!")
    elif sampling_type == "distance_sampling10":
        positive_sampling_index_list, negative_sampling_index_list = distance_sampling10(knn, sampling_num)
    elif sampling_type == "distance_sampling11":
        positive_sampling_index_list, negative_sampling_index_list = distance_sampling11(knn, sampling_num, distance_list)
    elif sampling_type == "distance_sampling12":
        raise ValueError("12 is not supported!")
    elif sampling_type == "distance_sampling13":
        raise ValueError("13 is not supported!")
    elif sampling_type == "distance_sampling14":
        raise ValueError("14 is not supported!")
    elif sampling_type == "distance_sampling_mix1":
        positive_sampling_index_list, negative_sampling_index_list = distance_sampling_mix1(knn, distance_list, anchor_pos, sampling_num, anchor_length, length_list, epoch)
    elif sampling_type == "distance_sampling15":
        if "big_uniref" in my_config.my_dict["root_write_path"]:
            sampling_area_list = pickle.load(open("/home/czh/clustering/protein/code/python/Neuprotein/clustering/big_uniref/agglomerative_1000_100_4_None", "rb"))
        elif "big_uniprot" in my_config.my_dict["root_write_path"]:
            sampling_area_list = pickle.load(open("/home/czh/clustering/protein/code/python/Neuprotein/clustering/big_uniprot/agglomerative_1000_100_4_None", "rb"))
        elif "big_qiita" in my_config.my_dict["root_write_path"]:
            sampling_area_list = pickle.load(open("/home/czh/clustering/protein/code/python/Neuprotein/clustering/big_uniref/agglomerative_1000_100_4_None", "rb"))
        elif "big_rt988" in my_config.my_dict["root_write_path"]:
            sampling_area_list = pickle.load(open("", "rb"))
        elif "big_gen50ks" in my_config.my_dict["root_write_path"]:
            sampling_area_list = pickle.load(open("", "rb"))
        else:
            raise ValueError("dataset_type error")
        pos = random.randint(0, len(sampling_area_list) - 1)
        
        positive_sampling_index_list, negative_sampling_index_list = distance_sampling15(knn, sampling_num, distance_list, sampling_area_list[pos])
    else:
        raise ValueError("Sampling type is not supported!")
    return positive_sampling_index_list, negative_sampling_index_list


def distance_sampling(distance_list, sampling_num, type):
    weight_list = []
    for distance in distance_list:
        weight = np.exp(-distance)
        # weight = np.exp(-distance / 1400)
        weight_list.append(weight)
    
    if type == "negative":
        weight_list = np.ones_like(weight_list) - weight_list

    weight_ratio_list = weight_list / np.sum(weight_list)
    importance_list = []
    for i in range(len(weight_ratio_list)):
        importance_list.append(np.sum(weight_ratio_list[:i]))
    importance_list = np.array(importance_list)

    sample_index_list = []
    while len(sample_index_list) < sampling_num:
        random_value = np.random.uniform()
        tem_list = np.where(importance_list > random_value)[0]
        if len(tem_list) == 0:
            sample_index_list.append(len(distance_list) - 1)
        elif ((tem_list[0] - 1) not in sample_index_list):
            sample_index_list.append(tem_list[0] - 1)
    
    sorted_sample_index_list = []
    for i in sample_index_list:
        sorted_sample_index_list.append((i, weight_list[i]))
    sorted_sample_index_list = sorted(sorted_sample_index_list, key = lambda a: a[1], reverse = True)
    return [i[0] for i in sorted_sample_index_list]

def distance_sampling2(knn, sampling_num):
    positive_sample_index = []
    negative_sample_index = []
    for i in range(sampling_num):
        positive_sample_index.append(knn[i + 1])
        negative_sample_index.append(knn[-(i + 1)])
    return positive_sample_index, negative_sample_index

# random sampling
def distance_sampling3(knn, sampling_num, distance_list, pos_begin_pos = 0,
                                                         pos_end_pos = 999,
                                                         neg_begin_pos = 0,
                                                         neg_end_pos = 999):
    positive_sample_index = []
    negative_sample_index = []
    positive_knn_sample   = []
    negative_knn_sample   = []
    for i in range(sampling_num):
        sampling_begin_pos = pos_begin_pos
        sampling_end_pos   = pos_end_pos
        first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        while (distance_list[knn[first_num]] < 0.0001):
            first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        sampling_begin_pos = neg_begin_pos
        sampling_end_pos   = neg_end_pos
        second_num = random.randint(sampling_begin_pos, sampling_end_pos)
        while (distance_list[knn[second_num]] < 0.0001):
            second_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        while(second_num == first_num):
            second_num = random.randint(sampling_begin_pos, sampling_end_pos)
        if first_num > second_num:
            first_num, second_num = second_num, first_num
        positive_sample_index.append(knn[first_num])
        negative_sample_index.append(knn[second_num])
        positive_knn_sample.append(first_num)
        negative_knn_sample.append(second_num)
    return positive_sample_index, negative_sample_index

def distance_sampling4(distance_list, anchor_pos, sampling_num, anchor_length, length_list, std = 30.0):
    positive_sample_index = []
    negative_sample_index = []
    
    for i in range(sampling_num):
        sampling_length = np.random.normal(loc = anchor_length, scale = std)
        while sampling_length > 500 or sampling_length < 3:
            sampling_length = np.random.normal(loc = anchor_length, scale = std)
        max_interval = 10000000
        pos = 0
        for j, length in enumerate(length_list):
            if abs(length - sampling_length) < max_interval and anchor_pos != j:
                max_interval = abs(length - sampling_length)
                pos = j
        first_num = pos

        sampling_length = np.random.normal(loc = anchor_length, scale = std)
        while sampling_length > 500 or sampling_length < 3:
            sampling_length = np.random.normal(loc = anchor_length, scale = std)
        max_interval = 10000000
        pos = 0
        for j, length in enumerate(length_list):
            if abs(length - sampling_length) < max_interval and anchor_pos != j:
                max_interval = abs(length - sampling_length)
                pos = j
        second_num = pos
        if distance_list[first_num] > distance_list[second_num]:
            first_num, second_num = second_num, first_num
        positive_sample_index.append(first_num)
        negative_sample_index.append(second_num)
    return positive_sample_index, negative_sample_index

def distance_sampling5(distance_list, anchor_pos, sampling_num, anchor_length, length_list, length_interval = 30.0):
    positive_sample_index = []
    negative_sample_index = []
    sampling_pos_list = []
    
    for i, length in enumerate(length_list):
        if length >= anchor_length - length_interval and length <= anchor_length + length_interval and anchor_pos != i:
            sampling_pos_list.append(i)
    for i in range(sampling_num):
        sampling_begin_pos = 0
        sampling_end_pos   = len(sampling_pos_list) - 1
        first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        second_num = random.randint(sampling_begin_pos, sampling_end_pos)
        while(second_num == first_num):
            second_num = random.randint(sampling_begin_pos, sampling_end_pos)
    
        if distance_list[sampling_pos_list[first_num]] > distance_list[sampling_pos_list[second_num]]:
            first_num, second_num = second_num, first_num
        positive_sample_index.append(sampling_pos_list[first_num])
        negative_sample_index.append(sampling_pos_list[second_num])
    return positive_sample_index, negative_sample_index

def distance_sampling6(knn, sampling_num, distance_list):
    positive_sample_index = []
    negative_sample_index = []  
    positive_knn_sample   = []
    negative_knn_sample   = []

    break_point_value = 0.3

    for i in range(len(knn)):
        if distance_list[knn[i]] < break_point_value:
            break_point_pos = i
        else:
            break
    if break_point_pos == 0:
        break_point_pos = 1
    
    for i in range(sampling_num):
        sampling_begin_pos = 0
        sampling_end_pos   = break_point_pos
        first_num  = random.randint(sampling_begin_pos, sampling_end_pos)

        sampling_begin_pos = 0
        sampling_end_pos   = break_point_pos    
        second_num = random.randint(sampling_begin_pos, sampling_end_pos)
    
        if first_num > second_num:
            first_num, second_num = second_num, first_num
        positive_sample_index.append(knn[first_num])
        negative_sample_index.append(knn[second_num])
        positive_knn_sample.append(first_num)
        negative_knn_sample.append(second_num)
    return positive_sample_index, negative_sample_index

def distance_sampling7(knn, pred_distance_list):
    positive_sample_index = []
    negative_sample_index = []  
    positive_knn_sample   = []
    negative_knn_sample   = []

    k_pos = 5
    k_neg = 30
    first_num = knn[random.randint(1, k_pos)]
    pred_knn = np.argsort(pred_distance_list[:1499])
    second_num = pred_knn[k_neg]

    # first_num = knn[1]
    # second_num = pred_knn[1]

    positive_sample_index.append(first_num)
    negative_sample_index.append(second_num)
    positive_knn_sample.append(0)
    negative_knn_sample.append(0)
    
    return positive_sample_index, negative_sample_index

def distance_sampling8(sampling_num, pred_distance_list, distance_list, anchor_pos):
    positive_sample_index = []
    negative_sample_index = []  
    
    n = 10

    test_distance = []
    true_distance = []
    for j in range(0, 3000):
        if anchor_pos == j:
            continue
        test_distance.append((j, pred_distance_list[j]))
        true_distance.append((j, distance_list[j]))
    s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
    s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
    tem_top_list = [l[0] for l in s_test_distance[:n] if l[0] in [j[0] for j in s_true_distance[:n]]]
    test_id_list = []
    true_id_list = []
    for i in range(n):
        if s_test_distance[i][0] not in tem_top_list:
            test_id_list.append(s_test_distance[i][0])
        if s_true_distance[i][0] not in tem_top_list:
            true_id_list.append(s_true_distance[i][0])

    for i in range(sampling_num):
        positive_sample_index.append(true_id_list[i])
        negative_sample_index.append(test_id_list[i])
    return positive_sample_index, negative_sample_index

def distance_sampling9(knn, sampling_num, distance_list, labels, anchor_pos, labels_dict, negative_dict):
    positive_sample_index = []
    negative_sample_index = []  
    positive_knn_sample   = []
    negative_knn_sample   = []
    
    current_label = labels[anchor_pos]
    if anchor_pos == -1:
        return distance_sampling10(knn, sampling_num, distance_list, break_point_pos = 3)
    else:
        candidate_positive_list = labels_dict[current_label]
        candidate_negative_list = negative_dict[current_label]
        for i in range(sampling_num):
            first_num  = random.randint(0, len(candidate_positive_list) - 1)
            second_num = random.randint(0, len(candidate_positive_list) - 1)
            first_num  = candidate_positive_list[first_num]
            second_num = candidate_positive_list[second_num]
            while second_num == first_num:
                second_num = random.randint(0, len(candidate_positive_list) - 1)
                second_num = candidate_positive_list[second_num]
            if distance_list[first_num] > distance_list[second_num]:
                first_num, second_num = second_num, first_num
            positive_sample_index.append(first_num)
            negative_sample_index.append(second_num)
        return positive_sample_index, negative_sample_index, positive_knn_sample, negative_knn_sample

def distance_sampling10(knn, sampling_num, break_point_pos = 3.0):
    positive_sample_index = []
    negative_sample_index = []  
    positive_knn_sample   = []
    negative_knn_sample   = []

    for i in range(sampling_num):
        sampling_begin_pos = 1
        sampling_end_pos   = break_point_pos
        first_num  = random.randint(sampling_begin_pos, sampling_end_pos)

        sampling_begin_pos = break_point_pos + 1
        sampling_end_pos   = 2999
        second_num = random.randint(sampling_begin_pos, sampling_end_pos)
        
        while(second_num == first_num):
            second_num = random.randint(sampling_begin_pos, sampling_end_pos)
    
        if first_num > second_num:
            first_num, second_num = second_num, first_num
        positive_sample_index.append(knn[first_num])
        negative_sample_index.append(knn[second_num])
        positive_knn_sample.append(first_num)
        negative_knn_sample.append(second_num)
    return positive_sample_index, negative_sample_index

def distance_sampling11(knn, sampling_num, distance_list, break_point_value = 0.3):
    positive_sample_index = []
    negative_sample_index = []  
    positive_knn_sample   = []
    negative_knn_sample   = []

    for i in range(len(knn)):
        if distance_list[knn[i]] < break_point_value:
            break_point_pos = i
        else:
            break


    if break_point_pos == 0:
        break_point_pos = 1
    
    for i in range(sampling_num):
        sampling_begin_pos = 1
        sampling_end_pos   = break_point_pos
        first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        sampling_begin_pos = break_point_pos + 1
        sampling_end_pos   = 2999
        second_num = random.randint(sampling_begin_pos, sampling_end_pos)
        
        while(second_num == first_num):
            second_num = random.randint(sampling_begin_pos, sampling_end_pos)
    
        if first_num > second_num:
            first_num, second_num = second_num, first_num
        positive_sample_index.append(knn[first_num])
        negative_sample_index.append(knn[second_num])
        positive_knn_sample.append(first_num)
        negative_knn_sample.append(second_num)
    return positive_sample_index, negative_sample_index


def distance_sampling12(knn, distance_list, anchor_pos, sampling_num, anchor_length, length_list, epoch):
    positive_sample_index = []
    negative_sample_index = []
    sampling_pos_list = []
    
    if 0 < anchor_length <= 100:
        length_interval = 30
    elif 100 < anchor_length <= 200:
        length_interval = 35
    elif 200 < anchor_length <= 300:
        length_interval = 40
    elif 300 < anchor_length <= 400:
        length_interval = 45
    else:
        length_interval = 50
    
    

    for i, length in enumerate(length_list):
        if length >= anchor_length - length_interval and length <= anchor_length + length_interval and anchor_pos != i:
            sampling_pos_list.append(i)
    for i in range(sampling_num):
        sampling_begin_pos = 0
        sampling_end_pos   = len(sampling_pos_list) - 1
        first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        second_num = random.randint(sampling_begin_pos, sampling_end_pos)
        while(second_num == first_num):
            second_num = random.randint(sampling_begin_pos, sampling_end_pos)
    
        if distance_list[sampling_pos_list[first_num]] > distance_list[sampling_pos_list[second_num]]:
            first_num, second_num = second_num, first_num
        positive_sample_index.append(sampling_pos_list[first_num])
        negative_sample_index.append(sampling_pos_list[second_num])
    return positive_sample_index, negative_sample_index

# mix
def distance_sampling13(knn, distance_list, anchor_pos, sampling_num, anchor_length, length_list, epoch):
    positive_sample_index = []
    negative_sample_index = []
    sampling_pos_list = []
    if epoch < 500:
        length_interval = 30
        for i, length in enumerate(length_list):
            if length >= anchor_length - length_interval and length <= anchor_length + length_interval and anchor_pos != i:
                sampling_pos_list.append(i)
    else:
        length_interval = 50
        for i, length in enumerate(length_list):
            if length >= anchor_length - length_interval and length <= anchor_length + length_interval and anchor_pos != i:
                if distance_list[i] < 0.8:
                    sampling_pos_list.append(i)
    for i in range(sampling_num):
        sampling_begin_pos = 0
        sampling_end_pos   = len(sampling_pos_list) - 1
        if len(sampling_pos_list) == 0:
            positive_sample_index.append(anchor_pos)
            negative_sample_index.append(anchor_pos)
        else:
            first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
            second_num = random.randint(sampling_begin_pos, sampling_end_pos)
            while(second_num == first_num):
                if len(sampling_pos_list) == 1:
                    break
                second_num = random.randint(sampling_begin_pos, sampling_end_pos)
            if distance_list[sampling_pos_list[first_num]] > distance_list[sampling_pos_list[second_num]]:
                first_num, second_num = second_num, first_num
            positive_sample_index.append(sampling_pos_list[first_num])
            negative_sample_index.append(sampling_pos_list[second_num])
    return positive_sample_index, negative_sample_index

def distance_sampling14(knn, sampling_num, distance_list):
    positive_sample_index = []
    negative_sample_index = []  
    positive_knn_sample   = []
    negative_knn_sample   = []
    break_point_pos_list = [1, 5, 10, 50, 100, 500, 1000, 1500, 2000, 2500, 2999]
    for i in range(sampling_num):
        sampling_begin_pos = break_point_pos_list[i]
        sampling_end_pos   = break_point_pos_list[i + 1]
        first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        while (distance_list[knn[first_num]] < 0.001):
            first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        sampling_begin_pos = break_point_pos_list[i + 1]
        sampling_end_pos   = break_point_pos_list[i + 2]
        second_num = random.randint(sampling_begin_pos, sampling_end_pos)
        while (distance_list[knn[second_num]] < 0.001):
            second_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        while(second_num == first_num):
            second_num = random.randint(sampling_begin_pos, sampling_end_pos)
        if first_num > second_num:
            first_num, second_num = second_num, first_num
        positive_sample_index.append(knn[first_num])
        negative_sample_index.append(knn[second_num])
        positive_knn_sample.append(first_num)
        negative_knn_sample.append(second_num)
    return positive_sample_index, negative_sample_index, positive_knn_sample, negative_knn_sample


def distance_sampling15(knn, sampling_num, distance_list, poly):
    pos_list = [p[0] for p in poly]
    neg_list = [p[1] for p in poly]
    pb, pe, nb, ne = np.min(pos_list), np.max(pos_list), np.min(neg_list), np.max(neg_list)
    if pb == len(knn):
        pb = len(knn) - 1
    if pe == len(knn):
        pe = len(knn) - 1
    if nb == len(knn):
        nb = len(knn) - 1
    if ne == len(knn):
        ne = len(knn) - 1
    # print(pb, pe, nb, ne)
    positive_sample_index = []
    negative_sample_index = []
    positive_knn_sample   = []
    negative_knn_sample   = []
    for i in range(sampling_num):
        while True:
            sampling_begin_pos = pb
            sampling_end_pos   = pe
            first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
            while (distance_list[knn[first_num]] < 0.0001):
                first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
            sampling_begin_pos = nb
            sampling_end_pos   = ne
            second_num = random.randint(sampling_begin_pos, sampling_end_pos)
            while (distance_list[knn[second_num]] < 0.0001):
                second_num  = random.randint(sampling_begin_pos, sampling_end_pos)
            while(second_num == first_num):
                second_num = random.randint(sampling_begin_pos, sampling_end_pos)
            if first_num > second_num:
                first_num, second_num = second_num, first_num
            if function.is_in_poly([first_num, second_num], poly):
                break

        positive_sample_index.append(knn[first_num])
        negative_sample_index.append(knn[second_num])
        positive_knn_sample.append(first_num)
        negative_knn_sample.append(second_num)
    return positive_sample_index, negative_sample_index


def distance_sampling_mix1(knn, distance_list, anchor_pos, sampling_num, anchor_length, length_list, epoch, pred_distance_list = None):
    positive_sample_index = []
    negative_sample_index = []
    # seg_list = [1, 150, 300, 550, 800, 1150, 1499]
    pos_seg_list = [1, 300, 600, 900, 1200, 1499]

    tem_positive_sample_index, tem_negative_sample_index = distance_sampling3(knn, 1, distance_list, pos_begin_pos = 0, pos_end_pos = 100, neg_begin_pos = 0, neg_end_pos = 100)
    positive_sample_index.extend(tem_positive_sample_index)
    negative_sample_index.extend(tem_negative_sample_index)

    tem_positive_sample_index, tem_negative_sample_index = distance_sampling3(knn, 1, distance_list, pos_begin_pos = 500, pos_end_pos = 600, neg_begin_pos = 1000, neg_end_pos = 1100)
    positive_sample_index.extend(tem_positive_sample_index)
    negative_sample_index.extend(tem_negative_sample_index)

    tem_positive_sample_index, tem_negative_sample_index = distance_sampling3(knn, 1, distance_list, pos_begin_pos = 0, pos_end_pos = 100, neg_begin_pos = 500, neg_end_pos = 600)
    positive_sample_index.extend(tem_positive_sample_index)
    negative_sample_index.extend(tem_negative_sample_index)



    return positive_sample_index, negative_sample_index

