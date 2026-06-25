import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """

    if method == "mse":
        pred = torch.tensor(pred, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        loss = ((pred - target)**2).mean()
    elif(method == "cross_entropy"):
        logits = torch.tensor(pred, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.long)
        
        max_logits = logits.max(dim=1, keepdim=True).values
        log_sum_exp = max_logits + torch.log(torch.exp(logits - max_logits).sum(dim =1,keepdim=True))
        
        true_class_logits = logits.gather(dim=1, index = target.unsqueeze(1))

        loss = (log_sum_exp -true_class_logits).mean()

    elif(method == "huber"):
        pred = torch.tensor(pred, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        error = (pred - target)
        abs_error = torch.abs(error)
        loss = torch.where(abs_error<delta, 0.5*(error)**2, delta*(abs_error-0.5*delta)).mean()
    else:
        raise ValueError("Invalid Method")

    return loss.item()
