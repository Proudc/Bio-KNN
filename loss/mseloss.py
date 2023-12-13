import torch
import torch.nn as nn
from torch import autograd
from torch.nn import Module
from torch.nn import functional
from torch.nn import Parameter


class MSELoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(MSELoss, self).__init__()
        self.init_loss(0)
        self.pos_mse_loss = nn.MSELoss()
        self.neg_mse_loss = nn.MSELoss()
    
    def init_loss(self, epoch):
        self.positive_mse_loss_list = []
        self.negative_mse_loss_list = []
        self.cross_mse_loss_list    = []
        self.rank_loss_list         = []

    def forward(self, p_input, p_target, n_input, n_target):
        self.positive_mse_loss = self.pos_mse_loss(p_input.float(), p_target.float())
        self.negative_mse_loss = self.neg_mse_loss(n_input.float(), n_target.float())        
        loss = sum([self.positive_mse_loss, self.negative_mse_loss])
        # print(loss.item())
        return self.positive_mse_loss