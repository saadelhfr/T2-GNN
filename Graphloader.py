import torch 


"""
Takes as input a networkx Graoh and returns a list of subgraphs based on a sampling strategy 
the goal is to be able to calculate the adjancency matrix and the PPR matrix without problems with the Ram memory 

"""


class GraphLoader():
    # This class is used to load the graph from the pickle file and divide it into batches based on the strongly connected  componenets 
    pass