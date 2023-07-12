import torch 
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data ,  DataLoader 
import networkx as nx


"""
Takes as input a networkx Graoh and returns a list of subgraphs based on a sampling strategy 
the goal is to be able to calculate the adjancency matrix and the PPR matrix without problems with the Ram memory 

"""


class GraphLoader(DataLoader):
    def __init__(self , load_path , save_path ,sampling_strategy, device ,directed :bool = False) :
        self.load_path = load_path
        self.save_path = save_path
        self.sampling_strategy = sampling_strategy
        self.directed = directed
        self.device = device
        self.list_subgraphs_path = []
        

    def load_nx_graph(self): 
        self.graph=nx.read_gpickle(self.load_path)
        self.nbr_nodes = self.graph.number_of_nodes()

    def save_torch_geo_subgraph(self , subgraph , path):
        torch.save(subgraph , path)
        self.list_subgraphs_path.append(path)


    def sample_subgraphs(self):

        if self.sampling_strategy == "connected_components" : 
            assert self.directed == False , "Connected components sampling strategy is only available for undirected graphs"
            self.subgraphs = list(nx.connected_components(self.graph))
            self.nbr_nodes_subgraphs = [len(subgraph) for subgraph in self.subgraphs]
            self.nbr_subgraphs = len(self.subgraphs)

        else : 
            raise NotImplementedError("Sampling strategy not implemented")


