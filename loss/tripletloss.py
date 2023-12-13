import torch
import torch.nn as nn
import torch.nn.functional as functional


class TripletLoss(nn.Module):
    def __init__(self, epochs):
        super(TripletLoss, self).__init__()
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

    def separate_loss(self, n_target, p_target, p_input, n_input, np_input, np_target):
        threshold = n_target - p_target
        rank_loss_mlp = functional.relu(p_input - n_input)

        positive_mse_loss = (p_input - p_target) ** 2
        negative_mse_loss = (n_input - n_target) ** 2
        np_mse_loss       = (np_input - np_target) ** 2
        
        mse_loss = positive_mse_loss + negative_mse_loss + np_mse_loss
        return rank_loss_mlp, mse_loss 

    def forward(self,
                my_config,
                epoch,
                p_input,
                p_target,
                n_input,
                n_target,
                np_input,
                np_target,
                p_weights = None,
                n_weights = None):
        
        if epoch in self.Ls:
            self.mlp_r, self.m = self.Ls[epoch]

        #############################
        ######### MLP loss ##########
        #############################
        threshold = n_target - p_target
        rank_loss_mlp = functional.relu(p_input - n_input + threshold)

        positive_mse_loss = (p_input - p_target) ** 2
        negative_mse_loss = (n_input - n_target) ** 2
        np_mse_loss       = (np_input - np_target) ** 2
        
        if p_weights is not None:
            positive_mse_loss *= p_weights
        if n_weights is not None:
            negative_mse_loss *= n_weights

        mse_loss = positive_mse_loss + negative_mse_loss + np_mse_loss

        return torch.mean(self.mlp_r * rank_loss_mlp),\
               torch.mean(self.m * torch.sqrt(mse_loss)),\
               torch.mean(self.mlp_r * rank_loss_mlp + self.m * torch.sqrt(mse_loss))

