import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w = torch.tensor(w_init, dtype=torch.float32,requires_grad=True)
    last_avg_gradient = None

    for i, (x,target)  in enumerate(micro_batches):
        x_tensor = torch.tensor(x,dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        loss = (torch.dot(w, x_tensor) - target_tensor)**2

        loss.backward()

        if (i + 1)%accum_steps == 0 or (i + 1) == len(micro_batches):
            grad = w.grad.clone()
            avg_gradient = grad/accum_steps
            last_avg_gradient = avg_gradient
            with torch.no_grad():
                w -= lr * last_avg_gradient
            w.grad.zero_()
    return w.tolist(), last_avg_gradient.tolist()
