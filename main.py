from PPR_Matrix.ppr import topk_ppr_matrix
import torch
import numpy as np
from scipy.sparse import csr_matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


A = np.array([[0, 1, 0, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1],
                  [1, 0, 1, 0, 1, 0]])  # adjacency matrix

alpha = 0.15
eps = 0.1
idx = np.arange(A.shape[0]) # Indices of all nodes
A_sparse = csr_matrix(A)
P = topk_ppr_matrix(A_sparse, alpha, eps, idx, topk=5)

print(P)
