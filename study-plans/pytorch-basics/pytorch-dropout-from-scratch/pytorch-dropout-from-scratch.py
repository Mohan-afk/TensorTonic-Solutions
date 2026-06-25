import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        if self.training == False or self.p == 0:
            return x
        if self.p == 1:
            return torch.zeros_like(x)
        
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
        y = (mask*x) / (1-self.p)
        return y
