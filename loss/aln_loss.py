import torch
import torch.nn as nn
import torch.nn.functional as functional


class AlnLoss(nn.Module):
    def __init__(self, epochs):
        super(AlnLoss, self).__init__()
        self.mlp_r, self.cnn_r, self.m, self.a = 1, 0, 1.0, 1.0

        step = epochs // 5
        self.Ls = {
            step * 0: (0, 0, 1, 0),
            step * 1: (1, 0, 1, 0),
            step * 2: (10, 0, 1, 0),
            step * 3: (50, 0, 1, 0),
            step * 4: (100, 0, 1, 0),
        }

        self.init_loss(0)
    
    def init_loss(self, epoch):
        self.positive_mse_loss_list = []
        self.negative_mse_loss_list = []
        self.cross_mse_loss_list    = []
        self.rank_loss_list         = []

    def forward(self,
                my_config,
                p_input,
                p_target,
                n_input,
                n_target,
                p_n_target,
                epoch,
                p_aln_input,
                n_aln_input,
                p_aln_target,
                n_aln_target, p_weights = None, n_weights = None):
        
        if epoch in self.Ls:
            self.mlp_r, self.cnn_r, self.m, self.a = self.Ls[epoch]


        #############################
        ######### MLP loss ##########
        #############################
        threshold = n_target - p_target

        rank_loss_mlp = functional.relu(p_input - n_input + threshold)

        positive_mse_loss = (p_input - p_target) ** 2
        negative_mse_loss = (n_input - n_target) ** 2
        
        if p_weights is not None:
            positive_mse_loss *= p_weights
        if n_weights is not None:
            negative_mse_loss *= n_weights

        mse_loss = positive_mse_loss + negative_mse_loss
        
        
        #############################
        ######### CNN loss ##########
        #############################
        if my_config.my_dict["dataset_type"] == "protein":
            loss_list_length = 20
        elif my_config.my_dict["dataset_type"] == "dna":
            loss_list_length = 4
        else:
            raise ValueError("dataset_type error")

        aln_loss_list = []
        if "_sum" in my_config.my_dict["cnn_feature_distance_type"]:
            rank_loss_cnn     = functional.relu(p_aln_input - n_aln_input + threshold)
            positive_aln_loss = (p_aln_input - p_target)
            negative_aln_loss = (n_aln_input - p_target)
            aln_loss          = positive_aln_loss ** 2 + negative_aln_loss ** 2
        
        elif "_sep" in my_config.my_dict["cnn_feature_distance_type"]:
            tem_p_aln_input = p_aln_input.sum(dim = 1)
            tem_n_aln_input = n_aln_input.sum(dim = 1)
            # rank_list = []
            # for i in range(len(threshold)):
            #     # test interval
            #     # tem_threshold = threshold[i] * torch.ones_like(threshold)
            #     # test max
            #     # tem_threshold = torch.max(threshold) * torch.ones_like(threshold)
            #     # test same
            #     tem_threshold = threshold
            #     rank_list.append(tem_threshold)
            # threshold = torch.stack(rank_list)
            rank_loss_cnn = functional.relu(tem_p_aln_input - tem_n_aln_input + threshold)

            # threshold     = n_aln_target - p_aln_target
            # rank_loss_cnn = torch.mean(functional.relu(p_aln_input - n_aln_input + threshold))
            
            positive_aln_loss = (p_aln_input - p_aln_target)
            negative_aln_loss = (n_aln_input - n_aln_target)
            aln_loss          = positive_aln_loss ** 2 + negative_aln_loss ** 2
            # for i in range(loss_list_length):
            #     tem_loss = positive_aln_loss[:, i] ** 2 + negative_aln_loss[:, i] ** 2
            #     aln_loss_list.append(torch.mean(tem_loss))

            
            # positive_aln_loss = (tem_p_aln_input - p_target)
            # negative_aln_loss = (tem_n_aln_input - n_target)
            # cross_aln_loss    = (tem_p_aln_input - tem_n_aln_input)
            # aln_loss          = positive_aln_loss ** 2 + negative_aln_loss ** 2 + cross_aln_loss ** 2
            
        else:
            raise ValueError("cnn_feature_distance_type error")

        return self.mlp_r * torch.mean(rank_loss_mlp),\
               self.m * torch.mean(torch.sqrt(mse_loss)),\
               self.a * torch.mean(aln_loss),\
               aln_loss_list,\
               torch.mean(self.mlp_r * rank_loss_mlp + self.cnn_r * rank_loss_cnn + self.m * torch.sqrt(mse_loss)) + self.a * torch.mean(aln_loss)
               