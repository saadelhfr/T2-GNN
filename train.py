
import time
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from model import Teacher_Features , Teacher_Edge , Student


class Train : 
    def __init__(self , epochs ,  device  , num_nodes, input_dim , output_dim , hidden2 , nbr_clusters , dropout_rate=0.2 , lr=0.001):
        self.epochs = epochs
        self.device = device
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden2 = hidden2
        self.nbr_clusters = nbr_clusters
        self.dropout_rate = dropout_rate
        self.lr = lr

        self.feature_teacher = Teacher_Features(self.num_nodes , self.input_dim , self.output_dim , self.hidden2 , self.dropout_rate , self.device)
        self.edge_teacher = Teacher_Edge(self.num_nodes , self.input_dim , self.output_dim , self.hidden2 , self.dropout_rate , self.device)
        self.student = Student(self.num_nodes , self.input_dim , self.output_dim , self.hidden2 , self.dropout_rate , self.device)

        self.feature_teacher.to(self.device)
        self.edge_teacher.to(self.device)
        self.student.to(self.device)

        self.feat_teach_optimizer = optim.Adam(self.feature_teacher.parameters() , lr=self.lr)
        self.edge_teach_optimizer = optim.Adam(self.edge_teacher.parameters() , lr=self.lr)
        self.student_optimizer = optim.Adam(self.student.parameters() , lr=self.lr)

        

        





        
