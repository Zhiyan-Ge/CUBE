import torch


def _assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        finite = tensor[torch.isfinite(tensor)]
        finite_min = finite.min().item() if finite.numel() else float("nan")
        finite_max = finite.max().item() if finite.numel() else float("nan")
        raise FloatingPointError(
            f"Non-finite values detected in {name}: "
            f"shape={tuple(tensor.shape)}, finite_min={finite_min}, finite_max={finite_max}"
        )


@torch.no_grad()
def batch_pearson(pred, target, eps=1e-12):
    """Per-sample, per-channel full-resolution Pearson. Returns [B, 3].

    Constant channels are assigned 0 correlation. NaN/Inf predictions are treated
    as a hard error rather than being silently converted to zero.
    """
    pred = pred.float()
    target = target.float()
    _assert_finite("Pearson prediction", pred)
    _assert_finite("Pearson target", target)

    pred = pred.flatten(2)
    target = target.flatten(2)

    pred = pred - pred.mean(dim=2, keepdim=True)
    target = target - target.mean(dim=2, keepdim=True)

    numerator = (pred * target).sum(dim=2)
    pred_ss = pred.square().sum(dim=2)
    target_ss = target.square().sum(dim=2)
    denominator = torch.sqrt(pred_ss * target_ss)

    valid = denominator > eps
    corr = torch.zeros_like(numerator)
    corr[valid] = numerator[valid] / denominator[valid]
    _assert_finite("Pearson correlation", corr)
    return corr
