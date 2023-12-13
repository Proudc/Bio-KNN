import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import multiprocessing


# 该函数用于计算两两向量之间的欧式距离，结果与np.linalg.norm(a - b)计算的结果一样
# 主要用于两个numpy.array list两两计算各种距离
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

# 基于separate的距离计算
def l2_dist_separate(q: np.ndarray, x: np.ndarray, dataset_type):
    total_dis = np.zeros((len(q), len(x)))
    if dataset_type == "protein":
        length = 20
    elif dataset_type == "dna":
        length = 4
    else:
        raise ValueError("dataset_type error")
    tem_length = int(q.shape[1] / length)
    for i in range(length):
        tem_q = q[:, i * tem_length : (i + 1) * tem_length]
        tem_x = x[:, i * tem_length : (i + 1) * tem_length]
        total_dis += l2_dist(tem_q, tem_x)
    return total_dis

def hyperbolic_dist_separate(q: np.ndarray, x: np.ndarray, dataset_type):
    total_dis = np.zeros((len(q), len(x)))
    if dataset_type == "protein":
        length = 20
    elif dataset_type == "dna":
        length = 4
    else:
        raise ValueError("dataset_type error")
    tem_length = int(q.shape[1] / length)
    for i in range(length):
        total_dis += hyperbolic_dist(q[:, i * tem_length : (i + 1) * tem_length], x[:, i * tem_length : (i + 1) * tem_length])
    return total_dis


def cosine_dist(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == x.shape[1]

    qq = np.sum(q * q, axis=1) ** 0.5
    xx = np.sum(x * x, axis=1) ** 0.5
    q = q / qq[:, np.newaxis]
    x = x / xx[:, np.newaxis]
    return 1 - np.dot(q, x.T)


# 计算MSE RMSE MAE的值
def get_mse_metric(query_id_list, base_id_list, distance_matrix, test_distance_matrix, skip_equal = False):
    y_pred = []
    y_true = []
    for query_id in query_id_list:
        for base_id in base_id_list:
            if skip_equal and query_id == base_id:
                continue
            y_pred.append(test_distance_matrix[query_id][base_id])
            y_true.append(distance_matrix[query_id][base_id])
    MSE  = mean_squared_error(y_true, y_pred)
    RMSE = np.sqrt(MSE)
    MAE  = mean_absolute_error(y_true, y_pred)
    
    print('Mean Squared Error (MSE)             : \033[32;1m%.4f\033[0m' % (MSE))
    print('Squared Mean Squared Error (RMSE)    : \033[32;1m%.4f\033[0m' % (RMSE))
    print('Mean Absolute Error (MAE)            : \033[32;1m%.4f\033[0m' % (MAE))
    return MSE, RMSE, MAE
''' 
def topk_recall(query_id_list, base_id_list, distance_matrix, test_distance_matrix, skip_equal = False):
    # 计算Top-k Recall的值
    result = {}
    s_test_distance_dict = {}
    s_true_distance_dict = {}
    for query_id in query_id_list:
        test_distance = []
        true_distance = []
        for base_id in base_id_list:
            if skip_equal and query_id == base_id - 5000:
                continue
            test_distance.append((base_id, test_distance_matrix[query_id][base_id]))
            true_distance.append((base_id, distance_matrix[query_id][base_id]))
        # reverse = True  表示按照从大到小排序
        # reverse = false 表示按照从小到大排序
        s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
        s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
        s_test_distance_dict[query_id] = s_test_distance
        s_true_distance_dict[query_id] = s_true_distance

    top_test_num = [1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 10000]
    top_true_num = [1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 10000]
    print("Test \t ", end = "")
    for top_k in top_true_num:
        print("Top-%d\t" % (top_k), end="")
    print()

    for tem_test_num in top_test_num:
        top_count_list = [0 for i in range(len(top_true_num))]
        for query_id in query_id_list:
            s_test_distance = s_test_distance_dict[query_id]
            s_true_distance = s_true_distance_dict[query_id]
            for pos in range(len(top_true_num)):
                tem_true_num = top_true_num[pos]
                tem_top_list = [l[0] for l in s_test_distance[:tem_test_num] if l[0] in [j[0] for j in s_true_distance[:tem_true_num]]]
                if tem_test_num == 1 and tem_true_num == 1 and s_test_distance[0][0] != s_true_distance[0][0]:
                    top11_true_list = [s_true_distance[0][0]]
                    for id, value in s_true_distance:
                        if value == s_true_distance[0][1]:
                            top11_true_list.append(id)
                    if s_test_distance[0][0] in top11_true_list:
                        tem_top_list.append(s_test_distance[0][0])
                top_count_list[pos] += len(tem_top_list)
        print("%4d \t" % tem_test_num, end = "")
        for pos in range(len(top_true_num)):
            tem_recall = top_count_list[pos] / (len(query_id_list) * top_true_num[pos])
            print("\033[32;1m%.4f\033[0m \t" % (tem_recall), end="")
            result[str(tem_test_num) + "-" + str(top_true_num[pos])] = tem_recall
        print()
    return result
'''
def topk_recall(query_id_list, base_id_list, distance_matrix, test_distance_matrix, skip_equal = False):
    # 计算Top-k Recall的值
    top_test_num = [1, 5, 10, 50, 100, 500, 1000]
    top_true_num = [1, 5, 10, 50, 100, 500, 1000]
    end_pos = max(np.max(top_test_num), np.max(top_true_num))
    result = {}
    global s_test_id_dict
    global s_true_id_dict
    s_test_distance_dict = {}
    s_true_distance_dict = {}
    s_test_id_dict = {}
    s_true_id_dict = {}
    
    for query_id in query_id_list:
        test_distance = []
        true_distance = []
        for base_id in base_id_list:
            if skip_equal and query_id == base_id:
                continue
            test_distance.append((base_id, test_distance_matrix[query_id][base_id]))
            true_distance.append((base_id, distance_matrix[query_id][base_id]))
        # reverse = True  表示按照从大到小排序
        # reverse = false 表示按照从小到大排序
        s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
        s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
        s_test_distance_dict[query_id] = s_test_distance[:20]
        s_true_distance_dict[query_id] = s_true_distance[:20]
        s_test_id_dict[query_id] = [value[0] for value in s_test_distance[:end_pos]]
        s_true_id_dict[query_id] = [value[0] for value in s_true_distance[:end_pos]]
        
        

    print("Test \t ", end = "")
    for top_k in top_true_num:
        print("%6d\t" % (top_k), end="")
    print()

    tem_r = []
    cpu_num = 10
    with multiprocessing.Pool(processes=cpu_num) as pool:
        seg_length = int(len(query_id_list) / cpu_num)
        for i in range(cpu_num):
            tem_r.append(pool.apply_async(intersect, args=(query_id_list[i * seg_length:(i + 1) * seg_length], s_test_id_dict, s_true_id_dict, top_test_num, top_true_num)))
        pool.close()
        pool.join()
    top_count_list = np.zeros((len(top_test_num), len(top_true_num)))
    for value in tem_r:
        value = value.get()
        for i in range(len(top_test_num)):
            for j in range(len(top_true_num)):
                top_count_list[i][j] += value[i][j]
    top_1_counter = 0
    for query_id in query_id_list:
        s_test_distance = s_test_distance_dict[query_id]
        s_true_distance = s_true_distance_dict[query_id]
        if s_test_distance[0][0] != s_true_distance[0][0]:
            top11_true_list = [s_true_distance[0][0]]
            for id, value in s_true_distance:
                if value == s_true_distance[0][1]:
                    top11_true_list.append(id)
            if s_test_distance[0][0] in top11_true_list:
                top_1_counter += 1
        else:
            top_1_counter += 1
    top_count_list[0][0] = top_1_counter
    for i, tem_test_num in enumerate(top_test_num):
        print("%4d \t" % tem_test_num, end = "")
        for j, tem_true_num in enumerate(top_true_num):
            tem_recall = top_count_list[i][j] / (len(query_id_list) * tem_true_num)
            print("\033[32;1m%.4f\033[0m \t" % (tem_recall), end="")
            result[str(tem_test_num) + "-" + str(tem_true_num)] = tem_recall
        print()
    return result

def intersect(query_id_list, s_test_id_dict, s_true_id_dict, test_num_list, true_num_list):
    counter_list = np.zeros((len(test_num_list), len(true_num_list)))
    for i, tem_test_num in enumerate(test_num_list):
        for j, tem_true_num in enumerate(true_num_list):
            for query_id in query_id_list:
                test_id = s_test_id_dict[query_id][:tem_test_num]
                true_id = s_true_id_dict[query_id][:tem_true_num]
                counter_list[i][j] += len(np.intersect1d(test_id, true_id))
    return counter_list

        
def test_recall(query_id_list, base_id_list, distance_matrix, test_distance_matrix):
    # 计算Top-k Recall的值
    # top_test_num = [1, 2, 3, 4, 5, 10, 50, 100]
    # top_true_num = [1, 2, 3, 4, 5, 10, 50, 100]
    top_test_num = [1]
    top_true_num = [1]
    print("Test \t ", end = "")
    for top_k in top_true_num:
        print("Top-%d\t" % (top_k), end="")
    print()
    for tem_test_num in top_test_num:
        counter1 = []
        counter2 = []
        recall_list = []
        for query_id in query_id_list:
            test_distance = []
            true_distance = []
            for j in base_id_list:
                if query_id == j:
                    continue
                test_distance.append((j, test_distance_matrix[query_id][j]))
                true_distance.append((j, distance_matrix[query_id][j]))
            # reverse = True  表示按照从大到小排序
            # reverse = false 表示按照从小到大排序
            s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
            s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
            tem_num = 0
            for i in range(len(s_true_distance)):
                if s_true_distance[i][1] < 0.80:
                    tem_num += 1
            tem_top_list = [l[0] for l in s_test_distance[:tem_num] if l[0] in [j[0] for j in s_true_distance[:tem_num]]]
            counter1.append(len(tem_top_list))
            counter2.append(tem_num)
            tem_top_list = [l[0] for l in s_test_distance[:2] if l[0] in [j[0] for j in s_true_distance[:2]]]
            recall_list.append(len(tem_top_list) / 2)
            # print(query_id, len(tem_top_list), tem_num)
        print("%4d \t" % tem_test_num, end = "")
        for pos in range(len(top_true_num)):
            print("\033[32;1m%.4f\033[0m \t \033[32;1m%.4f\033[0m \t" % (np.sum(counter1), np.sum(counter2)), end="")
            # print('Top \033[32;1m %3d \033[0m Test, Top \033[32;1m %3d \033[0m True, Recall: \033[32;1m%.4f\033[0m' % (top_test_num[pos], top_true_num[pos], top_count_list[pos] / (test_seq_num * top_test_num[pos])))
        print()
        return counter1, counter2, recall_list
        
        
def intersect_sizes(true_list, test_list):
    return np.array([len(np.intersect1d(true_value, list(test_value))) for true_value, test_value in zip(true_list, test_list)])

        
def top1_recall_test(query_id_list, base_id_list, distance_matrix, test_distance_matrix):
    # 计算Top-k Recall的值
    top_test_num = [1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
    top_true_num = [1]
    end_pos = np.max(top_test_num)
    result = {}
    
    s_test_distance_dict = {}
    s_true_distance_dict = {}
    s_test_id_dict = {}
    s_true_id_dict = {}
    
    test_knn = np.empty(dtype=np.int32, shape=(len(query_id_list), len(base_id_list)))
    true_knn = np.empty(dtype=np.int32, shape=(len(query_id_list), len(base_id_list)))

    for i in range(len(query_id_list)):
        test_knn[i] = np.argsort(test_distance_matrix[i])
        true_knn[i] = np.argsort(distance_matrix[i])
    

    print("Test \t ", end = "")
    for top_k in top_true_num:
        print("%6d\t" % (top_k), end="")
    print()



    true_top_id    = true_knn[:, :1]
    intersect_list = [intersect_sizes(test_knn[:, :tem_test_num], true_top_id) for tem_test_num in top_test_num]
    recall_list    = [np.mean(tem_list) for tem_list in intersect_list]
    
    
    for i, tem_test_num in enumerate(top_test_num):
        print("%4d \t" % tem_test_num, end = "")
        for j, tem_true_num in enumerate(top_true_num):
            tem_recall = recall_list[i]
            print("\033[32;1m%.4f\033[0m \t" % (tem_recall), end="")
            result[str(tem_true_num) + "-" + str(tem_test_num)] = tem_recall
        print()
    return result    

def all_recall(query_id_list, base_id_list, distance_matrix, test_distance_matrix, skip_equal = False):
    # 计算Top-k Recall的值
    s_test_distance_dict = {}
    s_true_distance_dict = {}
    for query_id in query_id_list:
        test_distance = []
        true_distance = []
        for base_id in base_id_list:
            if skip_equal and query_id == base_id:
                continue
            test_distance.append((base_id, test_distance_matrix[query_id][base_id]))
            true_distance.append((base_id, distance_matrix[query_id][base_id]))
        # reverse = True  表示按照从大到小排序
        # reverse = false 表示按照从小到大排序
        s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
        s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
        s_test_distance_dict[query_id] = s_test_distance
        s_true_distance_dict[query_id] = s_true_distance

    top_test_num   = {}
    result         = []
    query_interval = 5
    for i in range(1, 500, query_interval):
        top_test_num[i] = [i]
        result.append(0)
    
    for tem_test_num in top_test_num.keys():
        top_true_num = top_test_num[tem_test_num]
        top_count_list = [0 for i in range(len(top_true_num))]
        for query_id in query_id_list:
            s_test_distance = s_test_distance_dict[query_id]
            s_true_distance = s_true_distance_dict[query_id]
            for pos in range(len(top_true_num)):
                tem_true_num = top_true_num[pos]
                tem_top_list = [l[0] for l in s_test_distance[:tem_test_num] if l[0] in [j[0] for j in s_true_distance[:tem_true_num]]]
                if tem_test_num == 1 and tem_true_num == 1 and s_test_distance[0][0] != s_true_distance[0][0]:
                    top11_true_list = [s_true_distance[0][0]]
                    for id, value in s_true_distance:
                        if value == s_true_distance[0][1]:
                            top11_true_list.append(id)
                    if s_test_distance[0][0] in top11_true_list:
                        tem_top_list.append(s_test_distance[0][0])
                top_count_list[pos] += len(tem_top_list)
        tem_recall = top_count_list[pos] / (len(query_id_list) * top_true_num[pos])
        result[int(tem_test_num / query_interval)] = tem_recall

    return result
  


def get_ndcg(query_id_list, base_id_list, distance_matrix, test_distance_matrix):
    def get_dcg(label_list):
        dcgsum = 0
        for i in range(len(label_list)):
            dcg = (2 ** label_list[i] - 1) / np.log2(i + 2)
            dcgsum += dcg
        return dcgsum
    top_n_list = [1, 5, 10, 50, 100, 500, 1500]
    for top_n in top_n_list:
        ndcg_list = []
        for query_id in query_id_list:
            test_distance = []
            true_distance = []
            for j in base_id_list:
                if query_id == j:
                    continue
                test_distance.append((j, test_distance_matrix[query_id][j]))
                true_distance.append((j, distance_matrix[query_id][j]))
            # reverse = True  表示按照从大到小排序
            # reverse = false 表示按照从小到大排序
            s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
            s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
            test_id_list = [t[0] for t in s_test_distance[:top_n]]
            true_id_list = [t[0] for t in s_true_distance[:top_n]]
            label_list = [1 - distance_matrix[query_id][j] for j in test_id_list]
            label_dcg = get_dcg(label_list)
            ideal_label_list = [1 - distance_matrix[query_id][j] for j in true_id_list]
            ideal_dcg = get_dcg(ideal_label_list)
            if ideal_dcg == 0:
                ndcg_list.append(0)
            else:
                ndcg_list.append(label_dcg / ideal_dcg)
        print('\033[0mNDCG@\033[32;1m%3d:\033[32;1m%.4f\033[0m' % (top_n, np.mean(ndcg_list)))

    
def get_mAP(query_id_list, base_id_list, distance_matrix, test_distance_matrix):
    # top_test_num = [1, 5, 10, 50, 100, 500, 1000]
    top_test_num = [100]
    for tem_test_num in top_test_num:
        total_value_list = []
        for query_id in query_id_list:
            test_distance = []
            true_distance = []
            for j in base_id_list:
                if query_id == j:
                    continue
                test_distance.append((j, test_distance_matrix[query_id][j]))
                true_distance.append((j, distance_matrix[query_id][j]))
            # reverse = True  表示按照从大到小排序
            # reverse = false 表示按照从小到大排序
            s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
            s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
            test_id_list = [t[0] for t in s_test_distance]
            true_id_list = [t[0] for t in s_true_distance]
            tem_list = []
            tem_pos = 1
            for i in range(tem_test_num):
                test_id = test_id_list[i]
                if test_id in true_id_list[:tem_test_num]:
                    tem_value = tem_pos / (i + 1)
                    tem_pos += 1
                    tem_list.append(tem_value)
            if len(tem_list) == 0:
                total_value_list.append(0)
            else:
                total_value_list.append(np.mean(tem_list))
        print('\033[32;1m%3d\033[0m mean Average Precision (mAP)   : \033[32;1m%.4f\033[0m' % (tem_test_num, np.mean(total_value_list)))


def get_mAP_new(query_id_list, base_id_list, distance_matrix, test_distance_matrix, skip_equal = False):
    # top_test_num = [1, 5, 10, 50, 100, 500, 1000]
    top_test_num = [10, 100, 500]
    for tem_test_num in top_test_num:
        total_value_list = []
        for query_id in query_id_list:
            test_distance = []
            true_distance = []
            for j in base_id_list:
                if skip_equal and query_id == j:
                    continue
                test_distance.append((j, test_distance_matrix[query_id][j]))
                true_distance.append((j, distance_matrix[query_id][j]))
            # reverse = True  表示按照从大到小排序
            # reverse = false 表示按照从小到大排序
            s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
            s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
            test_id_list = [t[0] for t in s_test_distance]
            true_id_list = [t[0] for t in s_true_distance]
            tem_list = []
            tem_pos = 1
            for i in range(len(test_id_list)):
                test_id = test_id_list[i]
                if test_id in true_id_list[:tem_test_num]:
                    tem_value = tem_pos / (i + 1)
                    tem_pos += 1
                    tem_list.append(tem_value)
                    if tem_pos == tem_test_num + 1:
                        break
            if len(tem_list) == 0:
                total_value_list.append(0)
            else:
                total_value_list.append(np.mean(tem_list))
        print('\033[32;1m%3d\033[0m mean Average Precision (mAP_NEW)   : \033[32;1m%.4f\033[0m' % (tem_test_num, np.mean(total_value_list)))

        
def get_mrr(query_id_list, base_id_list, distance_matrix, test_distance_matrix):
    mrr_list = []
    for query_id in query_id_list:
        test_distance = []
        true_distance = []
        for j in base_id_list:
            if query_id == j:
                continue
            test_distance.append((j, test_distance_matrix[query_id][j]))
            true_distance.append((j, distance_matrix[query_id][j]))
        # reverse = True  表示按照从大到小排序
        # reverse = false 表示按照从小到大排序
        s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
        s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
        test_id_list = [t[0] for t in s_test_distance]
        true_id_list = [t[0] for t in s_true_distance]
        index = test_id_list.index(true_id_list[0]) + 1
        mrr_list.append(1 / index)
    print('Mean Reciprocal Rank (MRR)          : \033[32;1m%.4f\033[0m' % (np.mean(mrr_list)))
    


### 用于MYCNN
def test_all(query_id_list, base_id_list, distance_matrix, test_distance_matrix):
    print("************************************************************************************")
    print("-------------------------------This is MYCNN test_all------------------------------")
    test_seq_num = len(query_id_list)
    print('Test on {} protein sequences'.format(test_seq_num))
    print('Test Range: {}-{}. Base Range: {}-{}.'.format(query_id_list[0], query_id_list[-1], base_id_list[0], base_id_list[-1]))
    # get_mse_metric(query_id_list, base_id_list, distance_matrix, test_distance_matrix, skip_equal = True)
    # get_mAP_new(query_id_list, base_id_list, distance_matrix, test_distance_matrix, skip_equal = True)
    topk_result = topk_recall(query_id_list, base_id_list, distance_matrix, test_distance_matrix, skip_equal = False)
    print("************************************************************************************")
    return topk_result

