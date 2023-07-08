
import time
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from model import Teacher_Features , Teacher_Edge , Student
from sklearn.cluster import KMeans , SpectralClustering , AgglomerativeClustering , DBSCAN , OPTICS , Birch
from utils import students_t_kernel_euclidean , student_t_kernel_cosine , generate_targer_distribution


class Train : 
    def __init__(self , epochs ,  device  , num_nodes, input_dim , output_dim , hidden2 , nbr_clusters , dropout_rate=0.2 , lr=0.001 , clustering_method='kmeans'):
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

        self.clustering_method = clustering_method

    def pretrain_feat_teacher(self , feat_epochs , data ) :
        adj = data.adj
        X = data.x
        self.feature_teacher.train()
        with torch.no_grad() : 
            feat_teacher_output , _ = self.feature_teacher(X)
        clustering = self.clustering_method(self.nbr_clusters , n_init="auto")
        cluster_ids = clustering.fit_predict(feat_teacher_output.cpu().detach().numpy())
        self.feature_teacher.Cluster_Layer = torch.tensor(clustering.cluster_centers_).to(self.device)
        self.feature_teacher.Cluster_Layer.requires_grad = False

        for epoch in range(feat_epochs):
            feat_teacher_output , _ = self.feature_teacher(X)
            """
            Kl divergence between the target distribution and the student t kernel
            """

            Q = students_t_kernel_euclidean(feat_teacher_output , self.feature_teacher.Cluster_Layer)
            P = generate_targer_distribution(Q)
            kl_loss = nn.KLDivLoss()(torch.log(Q) ,P.detach())
    
            mse_loss = 0
            if self.output_dim == self.input_dim :
                mse_loss = nn.MSELoss()(feat_teacher_output , X)
            loss = kl_loss + mse_loss
            self.feat_teach_optimizer.zero_grad()
            loss.backward()
            self.feat_teach_optimizer.step()
            print("Epoch : {} , Loss : {}".format(epoch , loss.item()))

    def pretrain_edge_teacher(self , edge_epochs , data) :
        self.edge_teacher.train()
        adj = data.adj
        with torch.no_grad() : 
            edge_teacher_output , _ = self.edge_teacher(adj)
        clustering = self.clustering_method(self.nbr_clusters , n_init="auto")
        cluster_ids = clustering.fit_predict(edge_teacher_output.cpu().detach().numpy())
        self.edge_teacher.Cluster_Layer = torch.tensor(clustering.cluster_centers_).to(self.device)
        self.edge_teacher.Cluster_Layer.requires_grad = False

        for epoch in range(edge_epochs):
            edge_teacher_output , _ = self.edge_teacher(adj)
            """
            Kl divergence between the target distribution and the student t kernel
            """
            Q = students_t_kernel_euclidean(edge_teacher_output , self.edge_teacher.Cluster_Layer)
            P = generate_targer_distribution(Q)
            kl_loss = nn.KLDivLoss()(torch.log(Q) ,P.detach())
    
            mse_loss = 0
            if self.output_dim == self.input_dim :
                mse_loss = nn.MSELoss()(edge_teacher_output , adj)
            loss = kl_loss + mse_loss
            self.edge_teach_optimizer.zero_grad()
            loss.backward()
            self.edge_teach_optimizer.step()
            print("Epoch : {} , Loss : {}".format(epoch , loss.item()))
        
        



            







        

        

        

        





        
