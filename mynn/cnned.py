import torch
import torch.nn as nn

import math

POOL = nn.AvgPool1d

class CNNED(nn.Module):
    def __init__(self,
                 input_size,
                 target_size,
                 batch_size,
                 sampling_num,
                 max_seq_length,
                 channel,
                 device):
        super(CNNED, self).__init__()
        self.input_size     = input_size
        self.target_size    = target_size
        self.max_seq_length = max_seq_length
        self.device         = device
        self.channel        = channel
        
        total_layers = int(math.log2(self.max_seq_length))
        self.conv = nn.Sequential(
            nn.Conv1d(1, self.channel, 3, 1, padding=1, bias=False),
            POOL(2),
        )
        for i in range(total_layers - 1):
            self.conv.add_module("conv{}".format(i + 1), nn.Conv1d(self.channel, self.channel, 3, 1, padding=1, bias=False))
            self.conv.add_module("pool{}".format(i + 1), POOL(2))
        
        self.flat_size = self.input_size * self.channel

        self.fc1 = nn.Linear(self.flat_size, self.flat_size)
        self.fc2 = nn.Linear(self.flat_size, self.target_size)


    def forward(self, inputs_array, input_lens):
        
        
        anchor_input     = torch.stack(inputs_array[0]).to(self.device)
        positive_input   = torch.stack(inputs_array[1]).to(self.device)
        negative_input   = torch.stack(inputs_array[2]).to(self.device)

        anchor_result   = self.encode(anchor_input)
        positive_result = self.encode(positive_input)
        negative_result = self.encode(negative_input)

        return anchor_result[0],  \
               positive_result[0],\
               negative_result[0],\
               anchor_result[1],  \
               positive_result[1],\
               negative_result[1]
 
    
    def encode(self, x, lens = None):
        
        seq_num = len(x)
        x = x.permute(0, 2, 1)
        
        x = x.contiguous().view(-1, 1, self.max_seq_length)
        x = self.conv(x)
        x = x.view(seq_num, self.flat_size)

        y = self.fc1(x)
        y = torch.relu(y)
        y = self.fc2(y)
        return [y, x]
