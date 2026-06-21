import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    funcs ={
        "zeros": torch.zeros,
         "ones":  torch.ones,
        "full": lambda s: torch.full(s,value)
    }
    return funcs[method](shape).tolist()