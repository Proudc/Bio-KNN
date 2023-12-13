import torch
import torch.nn as nn
import torch.nn.functional as functional

class MultiHeadLoss(nn.Module):
    def __init__(self, epochs):
        super(MultiHeadLoss, self).__init__()
        self.mlp_r, self.m = 1, 1
        
        step = epochs // 5
        self.Ls = {
            step * 0: (0, 10),
            step * 1: (10, 10),
            step * 2: (10, 1),
            step * 3: (5, 0.1),
            step * 4: (1, 0.01),
        }

        self.init_loss(0)
    
    def init_loss(self, epoch):
        pass

    def forward(self,
                my_config,
                epoch,
                p_input_list,
                p_target_list,
                n_input_list,
                n_target_list,
                np_input_list,
                np_target_list):
        
        if epoch in self.Ls:
            self.mlp_r, self.m = self.Ls[epoch]

        rank_loss_list = []
        mse_loss_list  = []
        loss_list      = []

        for i in range(len(p_input_list)):
            #############################
            ######### MLP loss ##########
            #############################
            threshold = n_target_list[i] - p_target_list[i]
            rank_loss_mlp = functional.relu(p_input_list[i] - n_input_list[i] + threshold)

            positive_mse_loss = (p_input_list[i] - p_target_list[i]) ** 2
            negative_mse_loss = (n_input_list[i] - n_target_list[i]) ** 2
            np_mse_loss       = (np_input_list[i] - np_target_list[i]) ** 2

            mse_loss = positive_mse_loss + negative_mse_loss + np_mse_loss

            rank_loss_list.append(torch.mean(self.mlp_r * rank_loss_mlp))
            mse_loss_list.append(torch.mean(self.m * torch.sqrt(mse_loss)))
            loss_list.append(torch.mean(self.mlp_r * rank_loss_mlp + self.m * torch.sqrt(mse_loss)))

        return rank_loss_list, mse_loss_list, loss_list

