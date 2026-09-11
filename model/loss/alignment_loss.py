import torch.nn.functional as F


def ur_alignment_loss(ur_a, ur_b):
    """Simple alignment loss for UR features. Computes the mean squared error between the pooled representations of two UR feature maps."""
    pooled_a = ur_a.mean(dim=(-2, -1))
    pooled_b = ur_b.mean(dim=(-2, -1))
    return F.mse_loss(pooled_a, pooled_b)