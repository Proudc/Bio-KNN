
import evaluate_metric.metric

import numpy as np
import pickle


def analyze(test_distance_matrix, distance_matrix, query_id_list, base_id_list):
    result = evaluate_metric.metric.test_all(query_id_list, base_id_list, distance_matrix, test_distance_matrix)
    return [result["1-1"] * 100,
            result["5-5"] * 100,
            result["10-10"] * 100,
            result["50-50"] * 100,
            result["100-100"] * 100,
            result["500-500"] * 100,
            result["1000-1000"] * 100,]

# dataset_name = "big_uniprot"


random_seed = 666
random_seed_list = [666, 555, 444]
area_path_list = ["agglomerative_1000_100_1_None",
                  "kmeans_1000_100_2", "spectral_1000_100_2", "agglomerative_1000_100_2_None",
                  "kmeans_1000_100_4", "spectral_1000_100_4", "agglomerative_1000_100_4_None",
                  "kmeans_1000_100_8", "spectral_1000_100_8", "agglomerative_1000_100_8_None",]
area_path_list = ["agglomerative_1000_100_1_None", "agglomerative_1000_100_2_None", "agglomerative_1000_100_4_None", "agglomerative_1000_100_8_None"]
# area_path_list = [""]
target_size_list = [128, 64, 64, 64, 32, 32, 32, 16, 16, 16]
target_size_list = [128, 64, 32, 16]
target_size_list = [32, 64]
dataset_name_list = ["big_uniprot", "big_uniref"]

result = [[], [], [], [], [], [], []]
for j, dataset_name in enumerate(dataset_name_list):
    similairty_path = "./" + dataset_name + "/test_similarity_matrix_result"
    distance_matrix = 1 - pickle.load(open(similairty_path, "rb")) / 100
    seq_length_list = pickle.load(open("./" + dataset_name + "/seq_length_list", "rb"))
    query_id_list   = [i for i in range(0, 1000)]
    base_id_list    = [i for i in range(0, len(seq_length_list) - 6000)]
    
    for i, area in enumerate(area_path_list):
        target_size = target_size_list[j]
        
        for random_seed in random_seed_list:
            train_flag = "MYCNN_Ablation_average_" + dataset_name + "_" + str(random_seed) + "_" + area + "_" + str(target_size) + "_999"
            path = "./" + dataset_name + "/feature_distance_dir/test_allfeature_" + train_flag
            print(path)
            test_distance_matrix = pickle.load(open(path, "rb"))
            
            tem_result = analyze(test_distance_matrix, distance_matrix, query_id_list, base_id_list)
            for i in range(len(tem_result)):
                result[i].append(tem_result[i])

            # train_flag = "MYCNN_EqualSize_" + dataset_name + "_" + str(random_seed) + "_" + area + "_256_999"
            # path = "/mnt/data_hdd1/czh/Neuprotein/" + dataset_name + "/feature_distance_dir/test_cnnfeature_" + train_flag
            # print(path)
            # test_distance_matrix = pickle.load(open(path, "rb"))
            # tem_result = analyze(test_distance_matrix, distance_matrix, query_id_list, base_id_list)
            # for i in range(len(tem_result)):
            #     result[i].append(tem_result[i])

for i in range(len(result)):
    print('Mean: {:.4f}, Var: {:.4f}'.format(np.mean(result[i]), np.var(result[i])))



