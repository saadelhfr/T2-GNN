import torch 
import torch.nn.functional as F 
from torch.nn import KLDivLoss



def kl_divergence(P , Q , mean : bool = True):
    """
    Computes the KL divergence between P and Q
    """
    if mean :
        return torch.mean(torch.sum(P * torch.log(P/Q) , dim=1)) 
    else :    
        return torch.sum(torch.sum(P* torch.log(P/Q) , dim=1)) 
    
def js_divergence(P , Q , mean : bool = True):
    M = 0.5 * (P + Q)
    if mean :
        return 0.5 * kl_divergence(P , M , mean) + 0.5 * kl_divergence
    else :
        return 0.5 * kl_divergence(P , M , mean) + 0.5 * kl_divergence
    


