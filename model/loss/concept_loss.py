import torch.nn.functional as F


def concept_loss(pred, target):
    """MSE loss for concept features."""
    return F.mse_loss(pred, target)