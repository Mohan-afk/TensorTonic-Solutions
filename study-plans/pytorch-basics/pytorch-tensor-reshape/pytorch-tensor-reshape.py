import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x =  torch.tensor(x,dtype=torch.float32)
    funcs = {
        "flatten": lambda s : torch.flatten(s),
        "squeeze": lambda s : s.squeeze(),
        "transpose": lambda s : s.permute(1,0)
    }
    return funcs[op](x).tolist()
