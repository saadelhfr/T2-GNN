import torch 
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data ,  DataLoader  , Dataset
import networkx as nx
import random 
import os 
from .utils import get_adjacency_matrix_torch


random.seed(42)


"""
Takes as input a networkx Graoh and returns a list of subgraphs based on a sampling strategy 
the goal is to be able to calculate the adjancency matrix and the PPR matrix without problems with the Ram memory 



"""


class GraphLoader(Dataset):
    def __init__(self , load_path , sampling_strategy, device ,  save_path :str ="/Data/Subgraphs/"  ,directed :bool = False) :
        self.load_path = load_path
        self.save_path = save_path
        self.sampling_strategy = sampling_strategy
        self.directed = directed
        self.device = device
        self.list_subgraphs_paths :list[str] = []


    def __len__(self):
        return len(self.list_subgraphs_paths)

    def __getitem__(self , idx: int):
        data =  torch.load(self.list_subgraphs_paths[idx])
        return data


    def load_nx_graph(self): 
        self.graph=nx.read_gpickle(self.load_path)
        self.nbr_nodes = self.graph.number_of_nodes()

    def save_torch_geo_subgraph(self , subgraph , path : str) :
        torch.save(subgraph , path)
        self.list_subgraphs_paths.append(path)

    def torch_graph_from_set(self , set ): 
        subgraph = self.graph.subgraph(set)
        adj = get_adjacency_matrix_torch(subgraph)
        adj_aug = None
        data = from_networkx(subgraph)
        data = data.to(self.device)
        return data


    def sample_subgraphs(self):

        if self.sampling_strategy == "connected_components" : 
            assert self.directed == False , "Connected components sampling strategy is only available for undirected graphs"
            self.subgraphs = list(nx.connected_components(self.graph))
            self.nbr_nodes_subgraphs = [len(subgraph) for subgraph in self.subgraphs]
            self.nbr_subgraphs = len(self.subgraphs)

        else : 
            raise NotImplementedError("Sampling strategy not implemented")
    
    def get_subgraphs_set(self , nbr_subgraphs_merged : int =1000 ,  save : bool = True) : 
        """
    
        """
        list_subgraphs  = self.subgraphs.copy()
        random.shuffle(list_subgraphs)
        connected_components_sublists = [list_subgraphs[i:i + nbr_subgraphs_merged] for i in range(0, len(list_subgraphs), nbr_subgraphs_merged)]
        connected_components_sublists_merged = [set().union(*connected_components_sublist) for connected_components_sublist in connected_components_sublists]
        return connected_components_sublists_merged










        


