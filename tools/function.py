import torch
import random
import numpy as np

from mynn.mycnn            import MYCNN
from mynn.cnned            import CNNED


from loss.mseloss           import MSELoss
from loss.tripletloss       import TripletLoss
from loss.aln_loss          import AlnLoss
from loss.multi_head        import MultiHeadLoss


def set_dataset(root_write_path):
    # 设置训练测试集
    if root_write_path in ["/mnt/data_hdd3/czh/Neuprotein/big_uniprot"]:
        train_set, query_set, base_set = 1000, 1000, 478741 - 2000
    elif root_write_path in ["/mnt/data_hdd3/czh/Neuprotein/big_uniprot_train"]:
        train_set, query_set, base_set = 1000, 1000, 478741 - 2000
    elif root_write_path in ["/mnt/data_hdd3/czh/Neuprotein/big_uniref"]:
        train_set, query_set, base_set = 1000, 1000, 399869 - 2000
    elif root_write_path in ["/mnt/data_hdd3/czh/Neuprotein/big_qiita"]:
        train_set, query_set, base_set = 1000, 1000, 30497 - 2000
    elif root_write_path in ["/mnt/data_hdd3/czh/Neuprotein/big_rt988"]:
        train_set, query_set, base_set = 1000, 1000, 10000 - 2000
    elif root_write_path in ["/mnt/data_hdd3/czh/Neuprotein/big_gen50ks"]:
        train_set, query_set, base_set = 1000, 1000, 50001 - 2000
    else:
        raise Exception("root_write_path error")
    return train_set, query_set, base_set


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pad_seq_list(seq_list, max_length, pad_value = 0.0):
    value = [1.0 * pad_value for i in range(len(seq_list[0][0]))]
    final_pad_seq_list = []
    for seq in seq_list:
        assert len(seq) <= max_length, "Sequence length {} is larger than max_length {}".format(len(seq), max_length)

        for j in range(max_length - len(seq)):
            seq.append(value)
        final_pad_seq_list.append(seq)
    return final_pad_seq_list


def train_pad_seq_list(train_list, max_length, add_column_influence_flag = True, pad_value = 0.0):
    final_pad_seq_list = []
    for i in range(len(train_list)):
        tem_pad_seq_list = pad_seq_list(train_list[i], max_length, add_column_influence_flag, pad_value)
        final_pad_seq_list.append(tem_pad_seq_list)
    return final_pad_seq_list



def is_in_poly(point, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = point
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in
    
def initialize_loss(my_dict):
    # 设置loss
    batch_size   = my_dict["batch_size"]
    sampling_num = my_dict["sampling_num"]
    epoch_num    = my_dict["epoch_num"]
    
    if my_dict["loss_type"] == "mse":
        my_loss = MSELoss(batch_size, sampling_num)
    elif my_dict["loss_type"] == "triplet":
        my_loss = TripletLoss(epoch_num)
    elif my_dict["loss_type"] == "multihead":
        my_loss = MultiHeadLoss(epoch_num)
    elif my_dict["loss_type"] == "alnloss":
        my_loss = AlnLoss(epoch_num)
    else:
        raise ValueError("Loss Type Error")
    print("Init {} Loss Done !!!".format(my_dict["loss_type"]))
    return my_loss


def initialize_model(my_dict, max_seq_length):

    input_size = my_dict["input_size"]
    target_size = my_dict["target_size"]
    batch_size = my_dict["batch_size"]
    sampling_num = my_dict["sampling_num"]
    device = my_dict["device"]
    channel = my_dict["channel"]
    head_num = my_dict["head_num"]
    # 初始化网络
    if my_dict["network_type"] == "MYCNN":
        my_net = MYCNN(input_size, target_size, batch_size, sampling_num, max_seq_length, channel, device, head_num)
    elif my_dict["network_type"] == "CNNED":
        my_net = CNNED(input_size, target_size, batch_size, sampling_num, max_seq_length, channel, device)
    else:
        raise ValueError("Network Type Error")
    print("Init {} Model Done !!!".format(my_dict["network_type"]))
    return my_net

if __name__ == '__main__':
    point = [3, 3]
    poly = [[0, 0], [7, 3], [8, 8], [5, 5]]
    print(is_in_poly(point, poly))
