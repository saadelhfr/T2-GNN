import torch 


def calculate_ppr_matrix(A,  device, k=10 ,  alpha=0.15, tol=1e-6):

    alpha = torch.tensor(alpha, dtype=torch.float, device=device)
    I = torch.eye(A.size(0), device=device)
    
    # Create a normalized adjacency matrix
    deg_inv_sqrt = torch.rsqrt(A.sum(dim=1))
    A_hat = A * deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)
    A_hat = A_hat + I  # add self-loops

    # Initial guess for P
    P = torch.full(A_hat.shape, 1/A.size(0), device=device)
    diff = float('inf')
    while diff > tol:
        new_P = (1 - alpha) * torch.mm(A_hat, P) + alpha * I
        diff = torch.abs(P - new_P).sum()
        P = new_P

    if k is not None:
        # Get the k most important weights
        _, indices = torch.topk(P.view(-1), k)
        P = torch.zeros_like(P)
        P.view(-1)[indices] = new_P.view(-1)[indices]


    return P
