import torch
import torch.nn as nn

import math

POOL = nn.AvgPool1d

class MYCNN(nn.Module):
    def __init__(self,
                 input_size,
                 target_size,
                 batch_size,
                 sampling_num,
                 max_seq_length,
                 channel,
                 device,
                 head_num):
        super(MYCNN, self).__init__()
        self.input_size     = input_size
        self.target_size    = target_size
        self.max_seq_length = max_seq_length
        self.device         = device
        self.channel        = channel
        self.head_num       = int(head_num)
        
        total_layers = int(math.log2(self.max_seq_length))
        self.conv = nn.Sequential(
            nn.Conv1d(1, self.channel, 3, 1, padding=1, bias=False),
            POOL(2),
        )
        for i in range(total_layers - 1):
            self.conv.add_module("conv{}".format(i + 1), nn.Conv1d(self.channel, self.channel, 3, 1, padding=1, bias=False))
            self.conv.add_module("pool{}".format(i + 1), POOL(2))
        

        self.flat_size = self.input_size * self.channel

        self.head = nn.ModuleList()
        for i in range(self.head_num):
            tem_head =  nn.ModuleList([nn.Linear(self.flat_size, self.flat_size),
                nn.ReLU(),
                nn.Linear(self.flat_size, self.target_size),])
            self.head.append(tem_head)
    

    

    def inference(self, x):
        seq_num = len(x)
        x = x.permute(0, 2, 1)
        
        x = x.contiguous().view(-1, 1, self.max_seq_length)
        x = self.conv(x)
        x = x.view(seq_num, self.flat_size)
        
        y_list = []
        for i in range(self.head_num):
            y = self.head[i][0](x)
            y = self.head[i][1](y)
            y = self.head[i][2](y)
            y_list.append(y)

        y_list.append(x)
        return y_list


    def forward(self, inputs_array):
        
        
        anchor_input_list, positive_input_list, negative_input_list = [], [], []
        for inputs in inputs_array:
            anchor_input_list.append(torch.stack(inputs[0]).to(self.device))
            positive_input_list.append(torch.stack(inputs[1]).to(self.device))
            negative_input_list.append(torch.stack(inputs[2]).to(self.device))
            # print(torch.stack(inputs[0]).size())
        anchor_result   = self.encode(anchor_input_list)
        positive_result = self.encode(positive_input_list)
        negative_result = self.encode(negative_input_list)

        return anchor_result, positive_result, negative_result


    def encode(self, x_list):
        y_list = []
        for i in range(len(x_list)):
            x = x_list[i]
            seq_num = len(x)
            x = x.permute(0, 2, 1)
        
            x = x.contiguous().view(-1, 1, self.max_seq_length)
            x = self.conv(x)
            x = x.view(seq_num, self.flat_size)
            y = self.head[i][0](x)
            y = self.head[i][1](y)
            y = self.head[i][2](y)
            y_list.append(y)
        
        return y_list