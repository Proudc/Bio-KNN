import os
import time
import random
import pickle
import numpy as np

import torch


from tools import sampling_methods
from tools import test_methods
from tools import function
from tools import lds
from tools import fds
from tools import torch_feature_distance


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class NeuProteinTrainer(object):
    def __init__(self, my_config):
        self.my_config = my_config

    def data_prepare(self,
                     vec_list_path,
                     num_list_path,
                     train_similar_matrix_path,
                     train_aln_alp_path,
                     test_similar_matrix_path,
                     train_vec_list_path = None,
                     train_num_list_path = None,
                     train_seq_length_list_path = None, 
                     train_similarity_list_path = None,):
        print("Start Data Prepare...")        
        # read onehot embedding
        self.vec_list        = pickle.load(open(vec_list_path, 'rb'))
        self.seq_length_list = [len(seq) for seq in self.vec_list]
        self.total_seq_num   = len(self.vec_list)
        self.max_seq_length  = max(self.seq_length_list)
        self.min_seq_length  = min(self.seq_length_list)
        if train_seq_length_list_path is not None:
            self.train_seq_length_list = pickle.load(open(train_seq_length_list_path, 'rb'))
            self.max_seq_length        = max(np.max(self.train_seq_length_list), self.max_seq_length)
            self.min_seq_length        = min(np.max(self.train_seq_length_list), self.min_seq_length)

        print("Total Protein Number: {}".format(self.total_seq_num))
        print("Max Sequence Length : {}".format(self.max_seq_length))
        print("Min Sequence Length : {}".format(self.min_seq_length))

        self.merge_seq_list = self.vec_list[:self.my_config.my_dict["train_set"]]
        
        # padding sequence
        self.seq_pad_list = function.pad_seq_list(self.merge_seq_list, self.max_seq_length, pad_value = 0.0)    
        
        # read similairty matrix and get distance matrix
        self.train_similarity_matrix = pickle.load(open(train_similar_matrix_path, 'rb'))
        # self.test_similarity_matrix  = pickle.load(open(test_similar_matrix_path, 'rb'))
        if self.my_config.my_dict["similarity_type"] == "needle":
            self.train_distance_matrix = 1 - self.train_similarity_matrix / 100  
            # self.test_distance_matrix  = 1 - self.test_similarity_matrix / 100  
        elif self.my_config.my_dict["similarity_type"] == "ed":
            self.train_distance_matrix = self.train_similarity_matrix / np.max(self.train_similarity_matrix)
            # self.test_distance_matrix  = self.test_similarity_matrix / np.max(self.test_similarity_matrix)
        else:
            raise ValueError("similarity_type is not correct")
        
        self.avg_distance = np.mean(self.train_distance_matrix)
        self.max_distance = np.max(self.train_distance_matrix)
        print("Train Matrix size : {}".format(self.train_similarity_matrix.shape))
        # print("Test  Matrix size : {}".format(self.test_similarity_matrix.shape))
        print("Train Avg Distance: {}".format(self.avg_distance))
        print("Train Max Distance: {}".format(self.max_distance))

        # assert self.train_similarity_matrix.shape[0] == self.train_similarity_matrix.shape[1] == self.my_config.my_dict["train_set"], "Train Similarity Matrix Shape is Not Equal"
        # assert self.test_similarity_matrix.shape[0]  == self.my_config.my_dict["query_set"], "Query Similarity Matrix Shape is Not Equal"
        # assert self.test_similarity_matrix.shape[1]  == self.my_config.my_dict["base_set"], "Base Similarity Matrix Shape is Not Equal"
        
        
        self.seq_train_num = int(self.my_config.my_dict["train_set"] * self.my_config.my_dict["train_ratio"])
        print("Num Of Train Seq Is: {}".format(self.seq_train_num))

        if self.my_config.my_dict["loss_type"] in ["alnloss"]:
            print("Current Loss Type Is %s, So Alp Matrix Is Needed, Begin Load Alp Matrix ..." % self.my_config.my_dict["loss_type"])
            self.aln_alp_distance = pickle.load(open(train_aln_alp_path, 'rb'))
            print("End Load Alp Matrix !!!")
        else:
            print("Current Loss Type Is %s, So Alp Matrix Is Not Needed" % self.my_config.my_dict["loss_type"])
            self.aln_alp_distance = None

        if train_seq_length_list_path == None:
            print("Generating Train Seq Length List...")
            self.train_length_list = []
            for i in range(self.seq_train_num):
                self.train_length_list.append(self.seq_length_list[:self.seq_train_num])
        else:
            print("Loading Train Seq Length List...")
            self.train_vec_list          = pickle.load(open(train_vec_list_path, 'rb'))
            self.seq_train_num           = len(self.train_vec_list)
            self.train_num_list          = pickle.load(open(train_num_list_path, 'rb'))

            self.train_length_list       = pickle.load(open(train_seq_length_list_path, "rb"))
            self.train_similarity_matrix = np.array(pickle.load(open(train_similarity_list_path, "rb")))
            self.train_distance_matrix   = 1 - self.train_similarity_matrix / 100
            # self.train_distance_matrix = 1 - np.exp(-self.train_similarity_matrix * self.train_similarity_matrix / 600 / 250)
    
            self.train_merge_seq_list = []
            for i in range(len(self.train_vec_list)):
                tem_seq_1 = []
                for j in range(len(self.train_vec_list[i])):
                    tem_seq = []
                    for k in range(len(self.train_vec_list[i][j])):
                        tem_sec_seq = []
                        for p in range(len(self.train_vec_list[i][j][k])):
                            tem_sec_seq.append(self.train_vec_list[i][j][k][p])
                        # tem_sec_seq.append(0.0)
                        # tem_sec_seq.append(self.train_num_list[i][j][k])

                        tem_seq.append(tem_sec_seq)
                    tem_seq_1.append(tem_seq)
                self.train_merge_seq_list.append(tem_seq_1)
            
            if self.my_config.my_dict["similarity_type"] == "needle" and self.my_config.my_dict["dataset_type"] == "protein":
                self.total_train_seq_pad_list = function.train_pad_seq_list(self.train_merge_seq_list, self.max_seq_length, add_column_influence_flag = True, pad_value = 0.0)
            else:
                self.total_train_seq_pad_list = function.train_pad_seq_list(self.train_merge_seq_list, self.max_seq_length, add_column_influence_flag = False, pad_value = 0.0)
            self.train_seq_list = []
            for i in range(self.seq_train_num):
                self.train_seq_list.append(self.total_train_seq_pad_list[i])
        
        # sort distance from small to large
        self.knn = np.empty(dtype=np.int32, shape=(self.seq_train_num, self.seq_train_num))
        for i in range(self.seq_train_num):
            self.knn[i] = np.argsort(self.train_distance_matrix[i][:self.seq_train_num])
                
        self.total_seq_pad_list = torch.tensor(self.seq_pad_list, dtype = torch.float32)
        print("The Size Of Total Seq Pad List Is: ", self.total_seq_pad_list.size())

        self.train_seq_list = []
        for i in range(self.seq_train_num):
            self.train_seq_list.append(self.total_seq_pad_list[:self.seq_train_num])
        self.train_origin_seq_list = self.total_seq_pad_list[:self.seq_train_num]
        
        print("End Data Prepare !!!")
        
    def generate_train_data(self,
                            batch_size,
                            train_origin_seq_list,
                            train_seq_list,
                            train_distance_matrix,
                            sampling_num,
                            sampling_type,
                            epoch,
                            sampling_area_list,
                            pred_distance_matrix = None):

        new_list = [[i, self.seq_length_list[i]] for i in range(len(train_origin_seq_list))]
        new_list = [x[0] for x in new_list]
        total_result = []
        for i in range(0, len(train_origin_seq_list), batch_size):
            anchor_input,       positive_input,       negative_input     = [], [], []
            anchor_input_len,   positive_input_len,   negative_input_len = [], [], []
            positive_distance,  negative_distance,    cross_distance     = [], [], []
            positive_weights,   negative_weights,     cross_weights,     = [], [], []
            positive_aln_dist,  negative_aln_dist,    cross_aln_dist     = [], [], []
            total_anchor_index, total_positive_index, total_negative_index     = [], [], []
            for j in range(batch_size):
                if i + j >= len(train_origin_seq_list):
                    break
                anchor_pos = new_list[(i + j)]

                # ablation study
                positive_sampling_index_list, negative_sampling_index_list = sampling_methods.distance_sampling15(self.knn[anchor_pos], sampling_num, train_distance_matrix[anchor_pos], sampling_area_list[(i + j) % len(sampling_area_list)])

                
                # normal
                # positive_sampling_index_list, negative_sampling_index_list = sampling_methods.main_triplet_selection(sampling_type, sampling_num, self.knn[anchor_pos], train_distance_matrix[anchor_pos], pred_distance_matrix, anchor_pos, self.seq_length_list[anchor_pos], self.seq_length_list[:self.seq_train_num], epoch, self.my_config)

                # cross distance
                for k in range(sampling_num):
                    cross_distance.append(train_distance_matrix[positive_sampling_index_list[k]][negative_sampling_index_list[k]])
                    cross_weights.append(0)

                # positive distance
                for positive_index in positive_sampling_index_list:
                    anchor_input.append(train_origin_seq_list[anchor_pos])
                    positive_input.append(train_seq_list[anchor_pos][positive_index])

                    anchor_input_len.append(self.seq_length_list[anchor_pos])
                    positive_input_len.append(self.train_length_list[anchor_pos][positive_index])

                    positive_distance.append(train_distance_matrix[anchor_pos][positive_index])

                    total_anchor_index.append(anchor_pos)
                    total_positive_index.append(positive_index)


                    if self.aln_alp_distance is not None:
                        positive_aln_dist.append(self.aln_alp_distance[anchor_pos][positive_index])
                    else:
                        positive_aln_dist.append(0.0)

                    if self.weights is not None:
                        tem_index = lds.get_bin_idx(train_distance_matrix[anchor_pos][positive_index])
                        positive_weights.append(self.weights[tem_index])
                    else:
                        positive_weights.append(1.0)


                # negative distance
                for negative_index in negative_sampling_index_list:
                    negative_input.append(train_seq_list[anchor_pos][negative_index])
                    
                    negative_input_len.append(self.train_length_list[anchor_pos][negative_index])

                    negative_distance.append(train_distance_matrix[anchor_pos][negative_index])

                    total_negative_index.append(negative_index)

                    if self.aln_alp_distance is not None:
                        negative_aln_dist.append(self.aln_alp_distance[anchor_pos][negative_index])
                    else:
                        negative_aln_dist.append(0.0)

                    if self.weights is not None:
                        tem_index = lds.get_bin_idx(train_distance_matrix[anchor_pos][negative_index])
                        negative_weights.append(self.weights[tem_index])
                    else:
                        negative_weights.append(1.0)
            tem_batch = ([anchor_input,      positive_input,     negative_input], 
                         [anchor_input_len,  positive_input_len, negative_input_len], 
                         [positive_distance, negative_distance,  cross_distance],
                         [positive_weights,  negative_weights,   cross_weights],
                         [positive_aln_dist, negative_aln_dist,  cross_aln_dist], 
                         [total_anchor_index, total_positive_index, total_negative_index])
            total_result.append(tem_batch)
        return total_result

    def generate_global_train_data(self,
                            batch_size,
                            train_origin_seq_list,
                            train_seq_list,
                            train_distance_matrix,
                            sampling_num,
                            sampling_type,
                            epoch,
                            sampling_area_list):

        new_list = [[i, self.seq_length_list[i]] for i in range(len(train_origin_seq_list))]
        new_list = [x[0] for x in new_list]
        total_result = []
        split_length = int(len(train_origin_seq_list) / len(sampling_area_list))
        for pos, area in enumerate(sampling_area_list):
            area_result = []
            if pos != len(sampling_area_list) - 1:
                begin_pos = split_length * pos
                end_pos   = split_length * (pos + 1)
            else:
                begin_pos = split_length * pos
                end_pos   = split_length * (pos + 1)
            for i in range(begin_pos, end_pos, batch_size):
                anchor_input,       positive_input,       negative_input     = [], [], []
                anchor_input_len,   positive_input_len,   negative_input_len = [], [], []
                positive_distance,  negative_distance,    cross_distance     = [], [], []
                positive_weights,   negative_weights,     cross_weights,     = [], [], []
                positive_aln_dist,  negative_aln_dist,    cross_aln_dist     = [], [], []
                total_anchor_index, total_positive_index, total_negative_index     = [], [], []
                for j in range(batch_size):
                    if i + j >= len(train_origin_seq_list):
                        break
                    anchor_pos = new_list[(i + j)]
                    
                    # positive_sampling_index_list, negative_sampling_index_list = sampling_methods.distance_sampling3(self.knn[anchor_pos], sampling_num, train_distance_matrix[anchor_pos], pos_begin_pos=area[0], pos_end_pos=area[1], neg_begin_pos=area[2], neg_end_pos=area[3])
                    positive_sampling_index_list, negative_sampling_index_list = sampling_methods.distance_sampling15(self.knn[anchor_pos], sampling_num, train_distance_matrix[anchor_pos], area)


                    # cross distance
                    for k in range(sampling_num):
                        cross_distance.append(train_distance_matrix[positive_sampling_index_list[k]][negative_sampling_index_list[k]])
                        cross_weights.append(0)

                    # positive distance
                    for positive_index in positive_sampling_index_list:
                        anchor_input.append(train_origin_seq_list[anchor_pos])
                        positive_input.append(train_seq_list[anchor_pos][positive_index])

                        anchor_input_len.append(self.seq_length_list[anchor_pos])
                        positive_input_len.append(self.train_length_list[anchor_pos][positive_index])

                        positive_distance.append(train_distance_matrix[anchor_pos][positive_index])

                        total_anchor_index.append(anchor_pos)
                        total_positive_index.append(positive_index)


                        if self.aln_alp_distance is not None:
                            positive_aln_dist.append(self.aln_alp_distance[anchor_pos][positive_index])
                        else:
                            positive_aln_dist.append(0.0)

                        if self.weights is not None:
                            tem_index = lds.get_bin_idx(train_distance_matrix[anchor_pos][positive_index])
                            positive_weights.append(self.weights[tem_index])
                        else:
                            positive_weights.append(1.0)


                    # negative distance
                    for negative_index in negative_sampling_index_list:
                        negative_input.append(train_seq_list[anchor_pos][negative_index])

                        negative_input_len.append(self.train_length_list[anchor_pos][negative_index])

                        negative_distance.append(train_distance_matrix[anchor_pos][negative_index])

                        total_negative_index.append(negative_index)

                        if self.aln_alp_distance is not None:
                            negative_aln_dist.append(self.aln_alp_distance[anchor_pos][negative_index])
                        else:
                            negative_aln_dist.append(0.0)

                        if self.weights is not None:
                            tem_index = lds.get_bin_idx(train_distance_matrix[anchor_pos][negative_index])
                            negative_weights.append(self.weights[tem_index])
                        else:
                            negative_weights.append(1.0)
                tem_batch = ([anchor_input,      positive_input,     negative_input], 
                             [anchor_input_len,  positive_input_len, negative_input_len], 
                             [positive_distance, negative_distance,  cross_distance],
                             [positive_weights,  negative_weights,   cross_weights],
                             [positive_aln_dist, negative_aln_dist,  cross_aln_dist], 
                             [total_anchor_index, total_positive_index, total_negative_index])
                area_result.append(tem_batch)
            total_result.append(area_result)
        return total_result

    # get embedding of old method
    def get_embeddings(self, my_net, test_batch):
        my_net.eval()

        # initialize hidden
        hidden_element = torch.zeros(test_batch, self.my_config.my_dict["target_size"]).to(self.my_config.my_dict["device"])
        if self.my_config.my_dict["network_type"] == "STDLSTM":
            hidden = (hidden_element, hidden_element)
        else:
            hidden = hidden_element
        
        out_embedding_list = []
        cnn_embedding_list = []
        start_embedding_time = time.time()
        with torch.no_grad():
            for i in range(0, self.total_seq_num, test_batch):
                input_tensor = self.total_seq_pad_list[i : i + test_batch].to(self.my_config.my_dict["device"])
                if self.my_config.my_dict["network_type"] == "MYMLP":
                    out_feature = my_net.encoder(input_tensor)
                elif self.my_config.my_dict["network_type"] in ["CNNED", "sense"]:
                    out_feature, cnn_feature = my_net.encode(input_tensor)
                elif self.my_config.my_dict["network_type"] == "MYCNN":
                    out_feature_list, cnn_feature = my_net.encode(input_tensor)
                elif self.my_config.my_dict["network_type"] == "MYTRANSFORMER" or self.my_config.my_dict["network_type"] == "NeuroTransformer":
                    out_feature = my_net.encoder([input_tensor, self.seq_length_list[i : i + test_batch]])
                else:
                    out_feature = my_net.encoder([input_tensor, self.seq_length_list[i : i + test_batch]], hidden)
                out_embedding_list.append(out_feature.data)
                if self.my_config.my_dict["network_type"] in ["CNNED", "MYCNN", "sense"]:
                    cnn_embedding_list.append(cnn_feature.data)
        end_embedding_time = time.time()
        out_embedding_list = torch.cat(out_embedding_list, dim = 0)
        if self.my_config.my_dict["network_type"] in ["CNNED", "MYCNN", "sense"]:
            cnn_embedding_list = torch.cat(cnn_embedding_list, dim = 0)
        print("Embedding time: {}, Size of train embedding list: {}".format(end_embedding_time - start_embedding_time,out_embedding_list.size()))
        return out_embedding_list.cpu().numpy(), cnn_embedding_list.cpu().numpy()
    
    # get embedding of out multihead method
    def get_embeddings_multi_head(self, my_net, head_num, test_batch):
        assert self.my_config.my_dict["network_type"] == "MYCNN", "NETWORK_TYPE MUST BE MYCNN"
        my_net.eval()
        embedding_list = [[] for i in range(head_num + 1)]
        start_embedding_time = time.time()
        with torch.no_grad():
            for i in range(0, self.total_seq_num, test_batch):
                input_tensor = self.total_seq_pad_list[i : i + test_batch].to(self.my_config.my_dict["device"])
                feature_list = my_net.inference(input_tensor)
                assert head_num + 1 == len(feature_list)
                for j in range(len(feature_list)):
                    embedding_list[j].append(feature_list[j].data)
        end_embedding_time = time.time()
        cat_embedding_list = []
        for i in range(len(embedding_list)):
            cat_embedding_list.append(torch.cat(embedding_list[i], dim = 0).cpu().numpy())
        print("Embedding Time: {}, Size of Embedding List: {}".format(end_embedding_time - start_embedding_time, cat_embedding_list[0].shape))
        return cat_embedding_list
    
    def train(self):
        my_net = function.initialize_model(self.my_config.my_dict, self.max_seq_length).to(self.my_config.my_dict["device"])
        my_loss = function.initialize_loss(self.my_config.my_dict).to(self.my_config.my_dict["device"])
        optimizer = torch.optim.Adam(my_net.parameters(), lr = self.my_config.my_dict["learning_rate"])
        
        sampling_area_list = pickle.load(open(self.my_config.my_dict["area_path"], "rb"))

        for epoch in range(self.my_config.my_dict["epoch_num"]):
            start_time = time.time()
            
            my_net.train()
            my_loss.init_loss(epoch)

            pred_distance_matrix = None

            train_data = self.generate_train_data(self.my_config.my_dict["batch_size"],
                                                  self.train_origin_seq_list,
                                                  self.train_seq_list,
                                                  self.train_distance_matrix,
                                                  self.my_config.my_dict["sampling_num"],
                                                  self.my_config.my_dict["sampling_type"],
                                                  epoch,
                                                  sampling_area_list,
                                                  pred_distance_matrix = pred_distance_matrix)
            
            total_sampling_index = [[], [], []]
            for i, batch in enumerate(train_data):

                inputs_array,  inputs_len_array, distance_array, \
                weights_array, aln_dist_array,   sampling_index = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

                # split aanchor positive and negative
                total_sampling_index[0].extend(sampling_index[0])
                total_sampling_index[1].extend(sampling_index[1])
                total_sampling_index[2].extend(sampling_index[2])
                
                anchor_embedding,     positive_embedding,     negative_embedding, \
                anchor_cnn_embedding, positive_cnn_embedding, negative_cnn_embedding = my_net(inputs_array, inputs_len_array)

                positive_distance_target = torch.tensor(distance_array[0]).to(self.my_config.my_dict["device"])
                negative_distance_target = torch.tensor(distance_array[1]).to(self.my_config.my_dict["device"])
                cross_distance_target    = torch.tensor(distance_array[2]).to(self.my_config.my_dict["device"])

                anchor_length            = torch.tensor(inputs_len_array[0]).to(self.my_config.my_dict["device"])
                positive_length          = torch.tensor(inputs_len_array[1]).to(self.my_config.my_dict["device"])
                negative_length          = torch.tensor(inputs_len_array[2]).to(self.my_config.my_dict["device"])
                
                positive_aln_dist_target = torch.tensor(aln_dist_array[0]).to(self.my_config.my_dict["device"])
                negative_aln_dist_target = torch.tensor(aln_dist_array[1]).to(self.my_config.my_dict["device"])
                cross_aln_dist_target    = torch.tensor(aln_dist_array[2]).to(self.my_config.my_dict["device"])
                
                # positive_weights         = torch.tensor(weights_array[0]).view((-1, 1)).to(self.my_config.my_dict["device"])
                # negative_weights         = torch.tensor(weights_array[1]).view((-1, 1)).to(self.my_config.my_dict["device"])
                # cross_weights            = torch.tensor(weights_array[2]).view((-1, 1)).to(self.my_config.my_dict["device"])
                
                if self.my_config.my_dict["FDS"]:
                    positive_embedding = self.fds.smooth(positive_embedding, positive_distance_target, epoch)
                    negative_embedding = self.fds.smooth(negative_embedding, negative_distance_target, epoch)
                    
                # distance of MLP
                positive_learning_distance, \
                negative_learning_distance, \
                cross_learning_distance = torch_feature_distance.all_feature_distance(self.my_config.my_dict["all_feature_distance_type"],
                                                                                      anchor_embedding,
                                                                                      positive_embedding,
                                                                                      negative_embedding,
                                                                                      self.my_config.my_dict["channel"])
                # distance of CNN
                positive_aln_learning, \
                negative_aln_learning, \
                cross_aln_learning = torch_feature_distance.cnn_feature_distance(self.my_config.my_dict["cnn_feature_distance_type"],
                                                                                 anchor_cnn_embedding,
                                                                                 positive_cnn_embedding,
                                                                                 negative_cnn_embedding,
                                                                                 self.my_config.my_dict["channel"])
                
                if self.my_config.my_dict["loss_type"] == "triplet":
                    rank_loss, mse_loss, loss = my_loss(self.my_config,
                                                        epoch,
                                                        positive_learning_distance,
                                                        positive_distance_target,
                                                        negative_learning_distance,
                                                        negative_distance_target,
                                                        cross_learning_distance,
                                                        cross_distance_target)
                else:
                    raise ValueError("Loss Type Error")
                
                optimizer.zero_grad()                
                
                loss.backward()
                optimizer.step()
            end_time = time.time()
            

            # write index to log
            if (epoch < 30):
                if not os.path.exists(self.my_config.my_dict["root_write_path"] + "/total_sampling_index/"):
                    os.mkdir(self.my_config.my_dict["root_write_path"] + "/total_sampling_index/")
                pickle.dump(total_sampling_index, open(self.my_config.my_dict["root_write_path"] + "/total_sampling_index/" + self.my_config.my_dict["train_flag"] + "_" + str(epoch), "wb"))

            # print loss to stdout
            if (epoch + 1) % self.my_config.my_dict["print_epoch"] == 0:
                epoch_num = self.my_config.my_dict["epoch_num"]
                if self.my_config.my_dict["loss_type"] == "triplet":
                    print('Print Epoch: [{}/{}], Rank Loss: {:.6f}, Mse Loss: {:.4f}, Total Loss: {:.4f}'.format(epoch, epoch_num, rank_loss.item(), mse_loss.item(), loss.item()))
                else:
                    raise ValueError("Loss Type Error")
              
            # write model to file
            if epoch == self.my_config.my_dict["epoch_num"] - 1:
                if not os.path.exists(self.my_config.my_dict["save_model_path"]):
                    os.mkdir(self.my_config.my_dict["save_model_path"])
                save_model_name = self.my_config.my_dict["save_model_path"] + "/" + self.my_config.my_dict["train_flag"]
                torch.save(my_net.state_dict(), save_model_name)


    def train_multi_head(self):
        # Model mycnn(multi-head)
        self.my_config.my_dict["network_type"] = "MYCNN"
        # Loss multihead
        self.my_config.my_dict["loss_type"]    = "multihead"
        sampling_area_list = pickle.load(open(self.my_config.my_dict["area_path"], "rb"))
        self.my_config.my_dict["head_num"] = len(sampling_area_list)

        my_net    = function.initialize_model(self.my_config.my_dict, self.max_seq_length).to(self.my_config.my_dict["device"])
        my_loss   = function.initialize_loss(self.my_config.my_dict).to(self.my_config.my_dict["device"])
        optimizer = torch.optim.Adam(my_net.parameters(), lr = self.my_config.my_dict["learning_rate"])

        for epoch in range(self.my_config.my_dict["epoch_num"]):
            
            my_net.train()
            my_loss.init_loss(epoch)
            
            train_data = self.generate_global_train_data(self.my_config.my_dict["batch_size"],
                                                         self.train_origin_seq_list,
                                                         self.train_seq_list,
                                                         self.train_distance_matrix,
                                                         self.my_config.my_dict["sampling_num"],
                                                         self.my_config.my_dict["sampling_type"],
                                                         epoch,
                                                         sampling_area_list)
            
            global_data = []

            for i in range(len(train_data[0])):
                tem_train_data = []
                for j in range(len(train_data)):
                    tem_train_data.append(train_data[j][i])
                global_data.append(tem_train_data)
            
            total_sampling_index = [[], [], []]
            for i, batch in enumerate(global_data):
                inputs_array, inputs_len_array, distance_array, weights_array, aln_dist_array, sampling_index = [], [], [], [], [], []
                for value in batch:
                    inputs_array.append(value[0])
                    inputs_len_array.append(value[1])
                    distance_array.append(value[2])
                    weights_array.append(value[3])
                    aln_dist_array.append(value[4])
                    sampling_index.append(value[5])
                
                anchor_embedding_list, positive_embedding_list, negative_embedding_list = my_net(inputs_array)

                positive_distance_target_list, negative_distance_target_list, cross_distance_target_list = [], [], []
                anchor_length_list,            positive_length_list,          negative_length_list       = [], [], []
                positive_aln_dist_target_list, negative_aln_dist_target_list, cross_aln_dist_target_list = [], [], []
                for j in range(len(inputs_array)):
                    positive_distance_target = torch.tensor(distance_array[j][0]).to(self.my_config.my_dict["device"])
                    negative_distance_target = torch.tensor(distance_array[j][1]).to(self.my_config.my_dict["device"])
                    cross_distance_target    = torch.tensor(distance_array[j][2]).to(self.my_config.my_dict["device"])

                    anchor_length            = torch.tensor(inputs_len_array[j][0]).to(self.my_config.my_dict["device"])
                    positive_length          = torch.tensor(inputs_len_array[j][1]).to(self.my_config.my_dict["device"])
                    negative_length          = torch.tensor(inputs_len_array[j][2]).to(self.my_config.my_dict["device"])

                    positive_aln_dist_target = torch.tensor(aln_dist_array[j][0]).to(self.my_config.my_dict["device"])
                    negative_aln_dist_target = torch.tensor(aln_dist_array[j][1]).to(self.my_config.my_dict["device"])
                    cross_aln_dist_target    = torch.tensor(aln_dist_array[j][2]).to(self.my_config.my_dict["device"])
                    
                    positive_distance_target_list.append(positive_distance_target)
                    negative_distance_target_list.append(negative_distance_target)
                    cross_distance_target_list.append(cross_distance_target)
                    
                    anchor_length_list.append(anchor_length)
                    positive_length_list.append(positive_length)
                    negative_length_list.append(negative_length)
                    
                    positive_aln_dist_target_list.append(positive_aln_dist_target)
                    negative_aln_dist_target_list.append(negative_aln_dist_target)
                    cross_aln_dist_target_list.append(cross_aln_dist_target)

                    total_sampling_index[0].extend(sampling_index[j][0])
                    total_sampling_index[1].extend(sampling_index[j][1])
                    total_sampling_index[2].extend(sampling_index[j][2])
                    
                positive_learning_distance_list, negative_learning_distance_list, cross_learning_distance_list = [], [], []
                for j in range(len(inputs_array)):
                    positive_learning_distance, \
                    negative_learning_distance, \
                    cross_learning_distance = torch_feature_distance.all_feature_distance(self.my_config.my_dict["all_feature_distance_type"],
                                                                                          anchor_embedding_list[j],
                                                                                          positive_embedding_list[j],
                                                                                          negative_embedding_list[j],
                                                                                          self.my_config.my_dict["channel"])    
                    positive_learning_distance_list.append(positive_learning_distance)
                    negative_learning_distance_list.append(negative_learning_distance)
                    cross_learning_distance_list.append(cross_learning_distance)
                if self.my_config.my_dict["loss_type"] == "multihead":
                    rank_loss_list, mse_loss_list, loss_list = my_loss(self.my_config,
                                                                       epoch,
                                                                       positive_learning_distance_list,
                                                                       positive_distance_target_list,
                                                                       negative_learning_distance_list,
                                                                       negative_distance_target_list,
                                                                       cross_learning_distance_list,
                                                                       cross_distance_target_list)
                else:
                    raise ValueError("Loss Type Error")
                
                optimizer.zero_grad()                
                
                for tem_loss in loss_list:
                    tem_loss.backward(retain_graph=True)
                optimizer.step()
            

            # write index to log
            if (epoch < 30):
                if not os.path.exists(self.my_config.my_dict["root_write_path"] + "/total_sampling_index/"):
                    os.mkdir(self.my_config.my_dict["root_write_path"] + "/total_sampling_index/")
                pickle.dump(total_sampling_index, open(self.my_config.my_dict["root_write_path"] + "/total_sampling_index/" + self.my_config.my_dict["train_flag"] + "_" + str(epoch), "wb"))

            # print loss to stdout
            if (epoch + 1) % self.my_config.my_dict["print_epoch"] == 0:
                epoch_num = self.my_config.my_dict["epoch_num"]
                if self.my_config.my_dict["loss_type"] == "multihead":
                    print('Print Epoch: [{}/{}], loss: {}'.format(epoch, epoch_num, loss_list[0].item()))
                else:
                    raise ValueError("Loss Type Error")
            
            if epoch == self.my_config.my_dict["epoch_num"] - 1:
                if not os.path.exists(self.my_config.my_dict["save_model_path"]):
                    os.mkdir(self.my_config.my_dict["save_model_path"])
                save_model_name = self.my_config.my_dict["save_model_path"] + "/" + self.my_config.my_dict["train_flag"]
                torch.save(my_net.state_dict(), save_model_name)
                
                
    def extract_feature_from_path(self):
        model_path = self.my_config.my_dict["save_model_path"] + "/" + self.my_config.my_dict["train_flag"]
        my_net = function.initialize_model(self.my_config.my_dict, self.max_seq_length).to(self.my_config.my_dict["device"])
        my_net.load_state_dict(torch.load(model_path))
        total_all_embeddings_list, total_cnn_embeddings_list = [], []
        begin_pos, end_pos = 0, 5000
        while True:
            self.merge_seq_list = self.vec_list[begin_pos:end_pos]
            self.seq_pad_list = function.pad_seq_list(self.merge_seq_list, self.max_seq_length, pad_value = 0.0)    
            self.total_seq_pad_list = torch.tensor(self.seq_pad_list, dtype = torch.float32)
            all_embeddings_list, cnn_embeddings_list = self.get_embeddings(my_net, test_batch = 128)
            
            
            total_all_embeddings_list.extend(all_embeddings_list)
            total_cnn_embeddings_list.extend(cnn_embeddings_list)
            if end_pos == len(self.vec_list):
                break
            begin_pos = end_pos
            end_pos += 5000

            if end_pos > len(self.vec_list):
                end_pos = len(self.vec_list)
        total_all_embeddings_list = np.array(total_all_embeddings_list)
        total_cnn_embeddings_list = np.array(total_cnn_embeddings_list)
        print(total_all_embeddings_list.shape)
        print(total_cnn_embeddings_list.shape)
        
        train_all_embeddings_list = total_all_embeddings_list[:self.my_config.my_dict["train_set"]]
        train_cnn_embeddings_list = total_cnn_embeddings_list[:self.my_config.my_dict["train_set"]]
        query_all_embeddings_list = total_all_embeddings_list[self.my_config.my_dict["train_set"] : self.my_config.my_dict["train_set"] + self.my_config.my_dict["query_set"]]
        query_cnn_embeddings_list = total_cnn_embeddings_list[self.my_config.my_dict["train_set"] : self.my_config.my_dict["train_set"] + self.my_config.my_dict["query_set"]]
        base_all_embeddings_list  = total_all_embeddings_list[self.my_config.my_dict["train_set"] + self.my_config.my_dict["query_set"]:]
        base_cnn_embeddings_list  = total_cnn_embeddings_list[self.my_config.my_dict["train_set"] + self.my_config.my_dict["query_set"]:]

        print(len(train_all_embeddings_list), len(query_all_embeddings_list), len(base_all_embeddings_list))
        train_cnn_feature_distance = test_methods.get_feature_distance(train_cnn_embeddings_list, train_cnn_embeddings_list, "cnnfeature", self.my_config.my_dict["dataset_type"], self.my_config)
        test_cnn_feature_distance  = test_methods.get_feature_distance(query_cnn_embeddings_list, base_cnn_embeddings_list, "cnnfeature", self.my_config.my_dict["dataset_type"], self.my_config)
        test_methods.test_all_log(total_cnn_embeddings_list, self.my_config, "cnnfeature", train_cnn_feature_distance, test_cnn_feature_distance, epoch = 999)
        
        train_all_feature_distance = test_methods.get_feature_distance(train_all_embeddings_list, train_all_embeddings_list, "allfeature", self.my_config.my_dict["dataset_type"], self.my_config)
        test_all_feature_distance  = test_methods.get_feature_distance(query_all_embeddings_list, base_all_embeddings_list, "allfeature", self.my_config.my_dict["dataset_type"], self.my_config)
        test_methods.test_all_log(total_all_embeddings_list, self.my_config, "allfeature", train_all_feature_distance, test_all_feature_distance, epoch = 999)
                

    def extract_feature_from_path_multi_head(self):
        model_path = self.my_config.my_dict["save_model_path"] + "/" + self.my_config.my_dict["train_flag"]
        self.my_config.my_dict["network_type"] = "MYCNN"
        self.my_config.my_dict["loss_type"]    = "multihead"        

        sampling_area_list = pickle.load(open(self.my_config.my_dict["area_path"], "rb"))
        self.my_config.my_dict["head_num"] = len(sampling_area_list)
        

        my_net = function.initialize_model(self.my_config.my_dict, self.max_seq_length).to(self.my_config.my_dict["device"])

        my_net.load_state_dict(torch.load(model_path))
        all_embeddings_list = [[] for i in range(self.my_config.my_dict["head_num"] + 1)]

        # self.vec_list = self.vec_list[:6666]
        begin_pos, end_pos = 0, 5000
        while True:
            self.merge_seq_list = self.vec_list[begin_pos:end_pos]
            self.seq_pad_list = function.pad_seq_list(self.merge_seq_list, self.max_seq_length, pad_value = 0.0)    
            self.total_seq_pad_list = torch.tensor(self.seq_pad_list, dtype = torch.float32)
            embeddings_list = self.get_embeddings_multi_head(my_net, self.my_config.my_dict["head_num"], test_batch = 128)
            for i in range(len(embeddings_list)):
                all_embeddings_list[i].extend(embeddings_list[i])
            if end_pos == len(self.vec_list):
                break
            begin_pos = end_pos
            end_pos += 5000

            if end_pos > len(self.vec_list):
                end_pos = len(self.vec_list)
        all_embeddings_list = [np.array(value) for value in all_embeddings_list]
        print(len(all_embeddings_list), len(all_embeddings_list[0]), len(all_embeddings_list[-1]))
        print(self.my_config.my_dict["train_set"], self.my_config.my_dict["query_set"], self.my_config.my_dict["base_set"])

        train_all_embeddings_list, query_all_embeddings_list, base_all_embeddings_list = [], [], []
        for tem_index in range(len(all_embeddings_list)):
            train_all_embeddings_list.append(all_embeddings_list[tem_index][:self.my_config.my_dict["train_set"]])
            query_all_embeddings_list.append(all_embeddings_list[tem_index][self.my_config.my_dict["train_set"] : self.my_config.my_dict["train_set"] + self.my_config.my_dict["query_set"]])
            base_all_embeddings_list.append(all_embeddings_list[tem_index][self.my_config.my_dict["train_set"] + self.my_config.my_dict["query_set"]:])
        print(len(query_all_embeddings_list), len(query_all_embeddings_list[0]), len(query_all_embeddings_list[-1]))
        test_feature_distance_list = []
        for tem_index in range(len(all_embeddings_list)):
            query_embeddings_list = query_all_embeddings_list[tem_index]
            base_embeddings_list  = base_all_embeddings_list[tem_index]
            if (tem_index != len(all_embeddings_list) - 1):
                flag = "allfeature"
            else:
                flag = "cnnfeature"
                # flag = "allfeature"
            tem_feature_distance = test_methods.get_feature_distance(query_embeddings_list, base_embeddings_list, flag, self.my_config.my_dict["dataset_type"], self.my_config)
            test_feature_distance_list.append(tem_feature_distance)
        
        test_all_feature_distance = np.zeros((test_feature_distance_list[0].shape[0], test_feature_distance_list[0].shape[1]))
        test_cnn_feature_distance = np.zeros((test_feature_distance_list[0].shape[0], test_feature_distance_list[0].shape[1]))
        for tem_index in range(len(all_embeddings_list)):
            test_all_feature_distance += test_feature_distance_list[tem_index]
            if tem_index != len(all_embeddings_list) - 1:
                test_cnn_feature_distance += test_feature_distance_list[tem_index]    
        concat_embedding = np.hstack((all_embeddings_list[0], all_embeddings_list[1]))
        concat_cnn_embedding = np.hstack((all_embeddings_list[0], all_embeddings_list[1]))
        for tem_pos in range(2, len(all_embeddings_list)):
            concat_embedding = np.hstack((concat_embedding, all_embeddings_list[tem_pos]))
            if tem_pos != len(all_embeddings_list) - 1:
                concat_cnn_embedding = np.hstack((concat_embedding, all_embeddings_list[tem_pos]))
        train_all_feature_distance = None
        test_methods.test_all_log(concat_embedding, self.my_config, "allfeature", train_all_feature_distance, test_all_feature_distance, epoch = 999)
        test_methods.test_all_log(concat_cnn_embedding, self.my_config, "cnnfeature", train_all_feature_distance, test_cnn_feature_distance, epoch = 999)
        
        
