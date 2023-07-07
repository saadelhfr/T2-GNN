import torch 
import torch.nn.functional as F

def students_t_kernel_euclidean(z1 : torch.tensor , z2 : torch.tensor , v=1.0):
    """
    Computes the students t kernel between z1 and z2
    we consider that the two tensors are of shape 
    (N , n_features)
    where N can be different for z1 and z2

    """
    z1 = z1.unsqueeze(1)
    z2 = z2.unsqueeze(0)
    
    Q = 1.0/ (1.0 + torch.sum(torch.pow(z1-z2 , 2) , dim=2)/v)
    Q = Q.pow((v+1.0)/2.0)
    Q = Q / torch.sum(Q, dim=1, keepdim=True)  
    return Q


def student_t_kernel_cosine(z1 : torch.tensor , z2 : torch.tensor , v=1.0):
    """
    Computes the students t kernel between z1 and z2
    we consider that the two tensors are of shape 
    (N , n_features)
    where N can be different for z1 and z2

    """
    z1 = F.normalize(z1 , p=2 , dim=1)
    z2 = F.normalize(z2 , p=2 , dim=1)

    z1 = z1.unsqueeze(1)
    z2 = z2.unsqueeze(0)

    cosine_sim = 1 - torch.sum(z1*z2 , dim=-1)

    Q = 1.0/ (1.0 + torch.sum((1-cosine_sim) , dim=-1)/v )

    Q = Q.pow((v+1.0)/2.0)
    Q = Q/torch.sum(Q , dim=1 , keepdim=True)


    return Q
def generate_targer_distribution(Q):
    """
    Generates the target distribution for the student t kernel  

    """

    P = (Q**2) / torch.sum(Q, dim=1, keepdim=True)
    P = P / torch.sum(P , dim=1 , keepdim=True)
    return P

    