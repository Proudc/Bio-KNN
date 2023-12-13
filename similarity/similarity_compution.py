import os
import pickle
import multiprocessing
import numpy as np
import time

from similarity.seq_similarity import cal_similarity_between_seq
from similarity.edit_distance  import cal_ed_between_seq

# import Levenshtein

alp_list = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
def cal_alp_distance(seq1, seq2):
    distance = [0 for i in range(len(alp_list))]
    for i in range(len(seq1)):
        c1 = seq1[i]
        c2 = seq2[i]
        if c1 == c2:
            continue
        if c1 == "-" and c2 != "-":
            distance[alp_list.index(c2)] += 1
        if c1 != "-" and c2 == "-":
            distance[alp_list.index(c1)] += 1
        if c1 != "-" and c2 != "-":
            distance[alp_list.index(c1)] += 0.5
            distance[alp_list.index(c2)] += 0.5
    if sum(distance) != 0:
        return [dist/len(seq1) for dist in distance]
    else:
        return distance

def cal_ed_alp_distance(seq1, seq2):
    distance = [0 for i in range(len(alp_list))]
    for i in range(len(seq1)):
        c1 = seq1[i]
        c2 = seq2[i]
        if c1 == c2:
            continue
        if c1 == "-" and c2 != "-":
            distance[alp_list.index(c2)] += 1
        if c1 != "-" and c2 == "-":
            distance[alp_list.index(c1)] += 1
        if c1 != "-" and c2 != "-":
            distance[alp_list.index(c1)] += 0.5
            distance[alp_list.index(c2)] += 0.5
    if sum(distance) != 0:
        return [dist for dist in distance]
    else:
        return distance


dna_alp_list = ["A", "C", "G", "T"]
def cal_dna_alp_distance(seq1, seq2):
    distance = [0 for i in range(len(dna_alp_list))]
    for i in range(len(seq1)):
        c1 = seq1[i]
        c2 = seq2[i]
        if c1 == c2:
            continue
        if c1 == "-" and c2 != "-":
            distance[dna_alp_list.index(c2)] += 1
        if c1 != "-" and c2 == "-":
            distance[dna_alp_list.index(c1)] += 1
        if c1 != "-" and c2 != "-":
            distance[dna_alp_list.index(c1)] += 0.5
            distance[dna_alp_list.index(c2)] += 0.5
    if sum(distance) != 0:
        return [dist/len(seq1) for dist in distance]
    else:
        return distance

def cdist(seq_list1, seq_list2, similarity_type, dataset_type):
    matrix = np.zeros((len(seq_list1), len(seq_list2)))
    aln_matrix = []
    for i, seq1 in enumerate(seq_list1):
        tem_aln_list = []
        for j, seq2 in enumerate(seq_list2):
            similarity, qaln, saln = cal_similarity_between_seq(seq1, seq2, similarity_type, dataset_type)
            matrix[i, j] = similarity
            # tem_aln_list.append([qaln, saln])
        # aln_matrix.append(tem_aln_list)
    return matrix, aln_matrix

def protein_all_similarity(i, seq, seq_list, similarity_type, dataset_type, root_write_path):
    print("Begin Compute Similarity of Seq {}".format(i))
    if not os.path.exists(root_write_path + "/similarity_matrix"):
        os.makedirs(root_write_path + "/similarity_matrix")
    if not os.path.exists(root_write_path + "/similarity_matrix/matrix_{}".format(i)):
        similarity_matrix, aln_matrix = cdist([seq], seq_list, similarity_type, dataset_type)
        pickle.dump(similarity_matrix, open(root_write_path + "/similarity_matrix/matrix_{}".format(i), 'wb'))
        # pickle.dump(aln_matrix, open(root_write_path + "/similarity_matrix/aln_matrix_{}".format(i), 'wb'))
    print("End Compute Similarity of Protein {}".format(i))

def protein_similarity_batch(seq_list, root_write_path, similarity_type, dataset_type, processors = 96):
    pool = multiprocessing.Pool(processes = processors)
    for i, seq in enumerate(seq_list):
        if i < 5000:
            candidate_seq_list = seq_list[:5000]
        if 5000 <= i < 6000:
            candidate_seq_list = seq_list[6000:]
        if i >= 6000:
            break
        if similarity_type == "needle":
            pool.apply_async(protein_all_similarity, (i, seq_list[i], candidate_seq_list, similarity_type, dataset_type, root_write_path))
        elif similarity_type == "ed":
            pool.apply_async(protein_all_ed_distance, (i, seq_list[i], candidate_seq_list, "ed", root_write_path))
        else:
            raise ValueError("similarity_type must be needle or ed")
    pool.close()
    pool.join()

def protein_similarity_combain(seq_num, similarity_type, dataset_type, root_write_path):
    if similarity_type == "ed":
        ed_distance_combain(seq_num, root_write_path)
    else:
        row, column = 5000, 5000
        similarity_matrix_list = []
        # aln_alp_matrix = np.zeros((row, column), dtype="object")  
        for i in range(row):
            tem_matrix_list = pickle.load(open(root_write_path + "/similarity_matrix/matrix_{}".format(i), 'rb'))
            # tem_aln_matrix_list = pickle.load(open(root_write_path + "/similarity_matrix/aln_matrix_{}".format(i), 'rb'))
            similarity_matrix_list.append(tem_matrix_list)
            # if dataset_type == "protein":
            #     aln_alp_matrix[i] = [cal_alp_distance(value[0], value[1]) for value in tem_aln_matrix_list[0]]
            # elif dataset_type == "dna":
            #     aln_alp_matrix[i] = [cal_dna_alp_distance(value[0], value[1]) for value in tem_aln_matrix_list[0]]
            # else:
            #     raise ValueError("dataset_type must be protein or dna")
        similarity_matrix = np.array(similarity_matrix_list).reshape(row, column)
        
        pickle.dump(similarity_matrix, open(root_write_path + "/train_similarity_matrix_result", 'wb'))
        # pickle.dump(aln_alp_matrix, open(root_write_path + "/train_aln_alp_matrix_result2", 'wb'))
        print("The shape of similarity_matrix is {}".format(similarity_matrix.shape))

        row, column = 1000, seq_num - 6000
        similarity_matrix_list = []
        # aln_alp_matrix = np.zeros((row, column), dtype="object")     
        for i in range(row):
            tem_matrix_list = pickle.load(open(root_write_path + "/similarity_matrix/matrix_{}".format(i + 5000), 'rb'))
            # tem_aln_matrix_list = pickle.load(open(root_write_path + "/similarity_matrix/aln_matrix_{}".format(i + 5000), 'rb'))
            similarity_matrix_list.append(tem_matrix_list)
            # if dataset_type == "protein":
            #     aln_alp_matrix[i] = [cal_alp_distance(value[0], value[1]) for value in tem_aln_matrix_list[0]]
            # elif dataset_type == "dna":
            #     aln_alp_matrix[i] = [cal_dna_alp_distance(value[0], value[1]) for value in tem_aln_matrix_list[0]]
            # else:
            #     raise ValueError("dataset_type must be protein or dna")
        similarity_matrix = np.array(similarity_matrix_list).reshape(row, column)
        
        pickle.dump(similarity_matrix, open(root_write_path + "/test_similarity_matrix_result", 'wb'))
        # pickle.dump(aln_alp_matrix, open(root_write_path + "/test_aln_alp_matrix_result2", 'wb'))
        print("The shape of similarity_matrix is {}".format(similarity_matrix.shape))
        




# 计算n条序列的编辑距离矩阵
def ed_distance_cdist(seq_list1, seq_list2):
    matrix = np.zeros((len(seq_list1), len(seq_list2)))
    aln_matrix = []
    for i, seq1 in enumerate(seq_list1):
        tem_aln_list = []
        for j, seq2 in enumerate(seq_list2):
            distance, qaln, saln = cal_ed_between_seq(seq1, seq2)
            matrix[i, j] = distance
            tem_aln_list.append([qaln, saln])
        aln_matrix.append(tem_aln_list)
    return matrix, aln_matrix

def protein_all_ed_distance(i, seq_list, type, root_write_path):
    print("Begin Compute ED Distance of Seq {}".format(i))
    if not os.path.exists(root_write_path + "/ed_distance_matrix"):
        os.makedirs(root_write_path + "/ed_distance_matrix")
    if not os.path.exists(root_write_path + "/ed_distance_matrix/matrix_{}".format(i)):
        distance_matrix, aln_matrix = ed_distance_cdist([seq_list[i]], seq_list)
        pickle.dump(distance_matrix, open(root_write_path + "/ed_distance_matrix/matrix_{}".format(i), 'wb'))
        pickle.dump(aln_matrix, open(root_write_path + "/ed_distance_matrix/aln_matrix_{}".format(i), 'wb'))
    print("End Compute ED Distance of Seq {}".format(i))
    
def ed_distance_combain(seq_num, root_write_path):
    distance_matrix_list = []
    aln_alp_matrix = np.zeros((seq_num, seq_num), dtype="object")
    for i in range(seq_num):
        tem_matrix_list = pickle.load(open(root_write_path + "/ed_distance_matrix/matrix_{}".format(i), 'rb'))
        distance_matrix_list.append(tem_matrix_list)
        tem_aln_matrix_list = pickle.load(open(root_write_path + "/ed_distance_matrix/aln_matrix_{}".format(i), 'rb'))
        aln_alp_matrix[i] = [cal_ed_alp_distance(value[0], value[1]) for value in tem_aln_matrix_list[0]]

    distance_matrix = np.array(distance_matrix_list).reshape(seq_num, seq_num)
    pickle.dump(distance_matrix, open(root_write_path + "/ed_distance_matrix_result", 'wb'))
    pickle.dump(aln_alp_matrix, open(root_write_path + "/ed_aln_alp_matrix_result2", 'wb'))
    print("The shape of ed_distance_matrix is {}".format(distance_matrix.shape))

