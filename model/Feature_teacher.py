import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
from torch.utils.checkpoint import checkpoint

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Teacher_Features(nn.Module):
    def __init__(self , nbr_nodes, in_channels , out_channels , hid_channels , dropout  , device):
        super(Teacher_Features , self).__init__()
        self.device=device
        self.dropout = dropout
        # create the importance features
        self.imp_features = nn.Parameter(torch.empty(size=(nbr_nodes , in_channels )))
        nn.init.xavier_uniform_(self.imp_features.data , gain=1.414)
        # create Layers
        self.linear1 = nn.Linear(in_channels , hid_channels)
        self.linear2 = nn.Linear(hid_channels , hid_channels)
        self.linear3 = nn.Linear(hid_channels , out_channels)
        # Initialize weights
        self.weights_init()
        self.to(self.device)

    def weights_init(self):
        for layer in self.modules():
            if isinstance(layer , nn.Linear):
                nn.init.kaiming_uniform_(layer.weight.data , mode='fan_in' , nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)            

    def forward(self , x,pe_feat):
        idx = pe_feat._indices()[1]
        imp_feat_reduced = self.imp_features[idx]
        nan_mask = torch.isnan(x)
        x[nan_mask] = imp_feat_reduced[nan_mask]
        x.to(self.device)
        middle_representation = []
        h1 = self.linear1(x)
        middle_representation.append(h1)
        h2 = F.dropout(h1 , p=self.dropout , training=self.training)
        h2 = F.leaky_relu(self.linear2(h2))
        middle_representation.append(h2)
        h3 = F.dropout(h2 , p=self.dropout , training=self.training)
        h3 = F.leaky_relu(self.linear3(h3))
        middle_representation.append(h3)

        return h3 , middle_representation
    