import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    funcs ={
        "relu": lambda t: torch.maximum(t, torch.zeros_like(t)),
        "sigmoid": lambda t: (1/(1+torch.exp(-t))),
        "tanh": lambda t: (torch.exp(t)-torch.exp(-t))/(torch.exp(t)+torch.exp(-t)),
        "leaky_relu": lambda t: torch.where(t>0, t,0.01*t)
    }
    return funcs[method](x).tolist()