import os
import math
import pickle
import numpy as np


from config.CONSTANT import alp_to_onehot_dict_2
from config.CONSTANT import alp_to_onehot_dict_3
from config.CONSTANT import dna_alp_to_onehot_dict
from config.CONSTANT import alp_to_onehot_dict_20
from config.CONSTANT import alp_to_onehot_dict_24
from config.CONSTANT import alp_to_onehot_dict_25

from similarity.similarity_compution import protein_similarity_batch
from similarity.similarity_compution import protein_similarity_combain


def cal_similarity_matrix(seq_path, root_write_path, similarity_type, dataset_type):
    seq_list = pickle.load(open(seq_path, 'rb'))
    seq_num = len(seq_list)

    protein_similarity_batch(seq_list, root_write_path, similarity_type, dataset_type, processors = 60)
    protein_similarity_combain(seq_num, similarity_type, dataset_type, root_write_path)



def map_alp_to_onehot(seq_list, map_dict):
    onehot_list = []
    for seq in seq_list:
        vec = []
        for alp in seq:
            vec.append(map_dict[alp])
        onehot_list.append(vec)
    return onehot_list


def read_dna_txt(txt_file_path):
    seq_list = []
    with open(txt_file_path, 'r') as f:
        for line in f:
            seq_list.append(line.strip())
    return seq_list

def read_fasta_path(fasta_file_path):
    seq_list = []
    sequence = ''
    with open(fasta_file_path) as protein:
        for line in protein:
            line = line.strip() # removes \n
            if line.startswith('>'):
                if sequence != '':
                    seq_list.append(sequence)
                sequence = ''
            else:
                sequence = sequence + line
        seq_list.append(sequence)
    return seq_list

def processing(seq_list, root_write_path, similarity_type, dataset_type):

    alp_list = []
    for seq in seq_list:
        for a in seq:
            if a not in alp_list:
                alp_list.append(a)
    alp_size = len(alp_list)

    pickle.dump(seq_list, open(root_write_path + '/seq_list', 'wb'))
    
    length_list = [len(seq) for seq in seq_list]
    pickle.dump(length_list, open(root_write_path + '/seq_length_list', 'wb'))
    print("Alp Size:   ", alp_size)
    print("Max Length: ", np.max(length_list))
    print("Min Length: ", np.min(length_list))
    print("Avg Length: ", np.mean(length_list))
    

    # get onehot embedding
    if alp_size == 2:
        map_dict = alp_to_onehot_dict_2
    if alp_size == 3:
        map_dict = alp_to_onehot_dict_3
    if alp_size == 4:
        map_dict = dna_alp_to_onehot_dict
    if alp_size == 20:
        map_dict = alp_to_onehot_dict_20
    if alp_size == 24:
        map_dict = alp_to_onehot_dict_24
    if alp_size == 25:
        map_dict = alp_to_onehot_dict_25
    
    onehot_list = map_alp_to_onehot(seq_list, map_dict)
    pickle.dump(onehot_list, open(root_write_path + '/onehot_list', 'wb'))

    # cal NW similarity    
    cal_similarity_matrix(root_write_path + '/seq_list', root_write_path, similarity_type, dataset_type)


if __name__ == "__main__":
    
    from config import config_cnned as config
    if os.path.exists(config.fasta_file_path):
        seq_list = read_fasta_path(config.fasta_file_path)
    elif os.path.exists(config.txt_file_path):
        seq_list = read_dna_txt(config.txt_file_path)
    else:
        raise ValueError('path not exists!!!')
    
    dataset_type    = config.dataset_type
    root_write_path = config.root_write_path
    similarity_type = config.similarity_type
    processing(seq_list, root_write_path, similarity_type, dataset_type)

