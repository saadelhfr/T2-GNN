import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv



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

    def forward(self , x):
        x = torch.where(torch.isnan(x) , self.imp_features , x)
        middle_representation = []
        h1 = self.linear1(x)
        middle_representation.append(h1)
        h2 = F.dropout(h1 , p=self.dropout , training=self.training)
        h2 = F.leaky_relu(self.linear2(h2))
        middle_representation.append(h2)
        h3 = F.dropout(h2 , p=self.dropout , training=self.training)
        h3 = F.leaky_relu(h3)
        middle_representation.append(h3)

        return h3 , middle_representation
    
# Teacher Edge 
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
        self.pe_feat = torch.FloatTensor(torch.eye(nbr_nodes)).to(self.device)

        # Initialize weights
        self.to(self.device)
    
    def forward(self , Adj):
        middle_representation = []
        x = self.linear(self.pe_feat)

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
    

class Student(nn.Module):
    def __init__(self , nbr_nodes , in_channels , out_channels , hid_channels, feat_hidd , str_hidd , dropout , device , tau=0.5):
        super(Student , self).__init__()
        self.nbr_nodes = nbr_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.device = device
        self.tau = tau
        self.hid_channels = hid_channels
        self.feat_hidd = feat_hidd
        self.str_hidd = str_hidd
        
        self.gcn1 = DenseSAGEConv(in_channels , hid_channels)
        self.gcn2 = DenseSAGEConv(hid_channels , hid_channels)
        self.gcn3 = DenseSAGEConv(hid_channels , out_channels)

        self.feat_student = nn.Linear(self.feat_hidd , self.hid_channels , bias=True)
        self.str_student = nn.Linear(self.str_hidd , self.hid_channels , bias=True)
        self.to(self.device)
      
    def forward(self , X  ,Adj ) : 
        imp = torch.zeros((self.nbr_nodes , self.in_channels)).to(self.device)
        X = torch.where(torch.isnan(X) , imp , X).to(self.device)
        middle_representation = []
        h1 = self.gcn1(X , Adj)
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
    
    def sim(self , z1 : torch.tensor , z2 : torch.tensor):
        z1 = F.normalize(z1 ,dim=1)
        z2 = F.normalize(z2 ,dim=1)
        return torch.mm(z1 , z2.t())
    def semi_loss(self , z1 : torch.tensor , z2 : torch.tensor):
        f = lambda x : torch.exp(x/self.tau)
        refl_sim = f(self.sim(z1 , z1))
        bet_sim = f(self.sim(z1 , z2))
        return -torch.log(
            bet_sim.diag() / (refl_sim.sum(1) + bet_sim.sum(1) - refl_sim.diag())
        )
    
    def loss(self , z_student : torch.tensor , z_struct : torch.tensor , z_feat : torch.tensor , mean:bool = True):
        Emb_student_1 = z_student[0].squeeze(0)
        Emb_student_2 = z_student[1].squeeze(0)
        Emb_student_3 = z_student[2].squeeze(0)

        Emb_struct_1 = self.str_student(z_struct[0].squeeze(0))
        Emb_struct_2 = self.str_student(z_struct[1].squeeze(0))
        Emb_struct_3 = self.str_student(z_struct[2].squeeze(0))

        Emb_feat_1 =  self.feat_student(z_feat[0].squeeze(0))
        Emb_feat_2 =  self.feat_student(z_feat[1].squeeze(0))
        Emb_feat_3 =  self.feat_student(z_feat[2].squeeze(0))


        feat_stu_1 = self.semi_loss(Emb_student_1 , Emb_feat_1)
        feat_stu_2 = self.semi_loss(Emb_student_2 , Emb_feat_2)
        feat_stu_3 = self.semi_loss(Emb_student_3 , Emb_feat_3)

        struct_stu_1 = self.semi_loss(Emb_student_1 , Emb_struct_1)
        struct_stu_2 = self.semi_loss(Emb_student_2 , Emb_struct_2)
        struct_stu_3 = self.semi_loss(Emb_student_3 , Emb_struct_3)

        feat_struct_1 = feat_stu_1.mean() if mean else feat_stu_1.sum()
        feat_struct_2 = feat_stu_2.mean() if mean else feat_stu_2.sum()
        feat_struct_3 = feat_stu_3.mean() if mean else feat_stu_3.sum()

        struct_struct_1 = struct_stu_1.mean() if mean else struct_stu_1.sum()
        struct_struct_2 = struct_stu_2.mean() if mean else struct_stu_2.sum()
        struct_struct_3 = struct_stu_3.mean() if mean else struct_stu_3.sum()

        loss_feat = feat_struct_1 + feat_struct_2 + feat_struct_3
        loss_struct = struct_struct_1 + struct_struct_2 + struct_struct_3

        return loss_feat , loss_struct

        

        




        



