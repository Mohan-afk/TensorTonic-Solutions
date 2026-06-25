import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    mean = X.mean(dim=0)
    var = torch.mean((X - mean)**2, dim=0)
    X_hat = (X - mean)/torch.sqrt(var + eps)
    Y = gamma*X_hat + beta
    return Y 
