import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    max_logits = logits.max(dim = 1,keepdim=True).values
    exp_shifted = torch.exp(logits - max_logits)
    sum_exp_shifted = exp_shifted.sum(dim=1,keepdim=True)

    probs = exp_shifted/sum_exp_shifted
    
    return probs
