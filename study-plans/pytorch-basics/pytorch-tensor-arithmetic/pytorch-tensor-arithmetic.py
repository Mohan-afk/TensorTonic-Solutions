import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    funcs ={
        "add": lambda x,y : torch.add(x,y),
        "multiply": lambda x,y: torch.mul(x, y),
        "matmul": lambda x,y: torch.matmul(x,y),
        "power" : lambda x,y: torch.pow(x,y),
        "max" : lambda x,y:torch.max(x,y)
    }
    return funcs[op](x,y).tolist()