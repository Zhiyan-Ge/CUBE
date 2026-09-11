import torch


@torch.no_grad()
def batch_pearson(pred, target, eps=1e-12):
    """Per-sample, per-channel full-resolution Pearson: returns [B,3]."""
    pred = pred.float().flatten(2)
    target = target.float().flatten(2)

    pred = pred - pred.mean(dim=2, keepdim=True)
    target = target - target.mean(dim=2, keepdim=True)

    numerator = (pred * target).sum(dim=2)
    denominator = torch.sqrt(
        pred.square().sum(dim=2) * target.square().sum(dim=2)
    )

    corr = numerator / denominator.clamp_min(eps)
    corr = torch.where(denominator > eps, corr, torch.zeros_like(corr))
    return corr
