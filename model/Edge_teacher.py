import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
from torch.utils.checkpoint import checkpoint

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Teacher_Edge(nn.Module):
    def __init__(self , nbr_nodes , in_channels , out_channels , hid_channels , dropout ,  device):
        super(Teacher_Edge , self).__init__()
        self.nbr_nodes = nbr_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.device = device
        self.gcn1 = DenseSAGEConv(in_channels , hid_channels)
        self.gcn2 = DenseSAGEConv(hid_channels , hid_channels)
        self.gcn3 = DenseSAGEConv(hid_channels , out_channels)
        self.linear = nn.Linear(self.nbr_nodes , self.in_channels  , bias=True )
 
        # Initialize weights
        self.to(self.device)
    
    def forward(self , Adj , pe_feat , X):
        middle_representation = []
        x = self.linear(pe_feat)
        mask = torch.isnan(X)
        X[mask] = x[mask]

        h1 = self.gcn1(x , Adj)
        middle_representation.append(h1)
        h1 = F.dropout(h1 , p=self.dropout , training=self.training)
        h1 = F.leaky_relu(h1)
        h2 = self.gcn2(h1 , Adj)
        middle_representation.append(h2)
        h2 = F.dropout(h2 , p=self.dropout , training=self.training)
        h2 = F.leaky_relu(h2)
        h3 = self.gcn3(h2 , Adj)
        middle_representation.append(h3)

        return h3 , middle_representation
    