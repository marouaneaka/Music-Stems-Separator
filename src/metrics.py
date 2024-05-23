import torch

def myMetric(y_, y, q, coarse=True):
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0,1)   # shape = [num_sources, batch_size, num_channels, chunk_size]
    if coarse:
        loss = torch.mean(loss, dim=(-1,-2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile 
    return (loss * mask).mean()