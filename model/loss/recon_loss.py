import torch
import torch.nn.functional as F


def pearson_loss(pred, target, eps=1e-8):
    """Differentiable per-sample Pearson loss for [B,1,H,W].

    The loss is computed independently per sample and per channel. Samples whose
    target variance is effectively zero are excluded from the batch mean so the
    model is not forced to invent variation on empty targets.
    """
    pred = pred.float()
    target = target.float()

    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)

    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)

    target_var = target_centered.square().sum(dim=1)
    valid = target_var > eps

    if not torch.any(valid):
        return pred.sum() * 0.0

    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=1) * target_var + eps
    )

    pearson = torch.zeros_like(target_var)
    pearson[valid] = numerator[valid] / denominator[valid]
    pearson = pearson.clamp(-1.0 + eps, 1.0 - eps)

    loss = 1.0 - pearson
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return loss[valid].mean()


def multi_scale_pearson_loss(pred, target, scales, scale_weights, eps=1e-8):
    """Multi-scale Pearson loss for single-channel inputs [B,1,H,W].

    scales: iterable of integers (1,2,4...) representing avg_pool downsampling factors
    scale_weights: same-length iterable of floats summing to 1 (not strictly required)
    Returns weighted sum of (1 - Pearson) at each scale.
    """
    if scales is None or scale_weights is None:
        # fallback to single-scale
        return pearson_loss(pred, target, eps=eps)

    total = 0.0
    for s, w in zip(scales, scale_weights):
        if s == 1:
            l = pearson_loss(pred, target, eps=eps)
        else:
            p = F.avg_pool2d(pred, kernel_size=s, stride=s)
            t = F.avg_pool2d(target, kernel_size=s, stride=s)
            l = pearson_loss(p, t, eps=eps)
        total = total + float(w) * l
    return total


def ssim_loss(pred, target, window_size=11):
    """SSIM loss, range [0,1]。"""
    p = window_size // 2
    mu_x = F.avg_pool2d(pred, window_size, 1, p)
    mu_y = F.avg_pool2d(target, window_size, 1, p)
    var_x = F.avg_pool2d(pred * pred, window_size, 1, p) - mu_x ** 2
    var_y = F.avg_pool2d(target * target, window_size, 1, p) - mu_y ** 2
    cov_xy = F.avg_pool2d(pred * target, window_size, 1, p) - mu_x * mu_y

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    )
    return 1 - ssim.mean()


def he_recon_loss(pred, target, alpha=1.0):
    """HE reconstruction: L1 + α·SSIM。"""
    return F.l1_loss(pred, target) + alpha * ssim_loss(pred, target)


def weighted_l1(pred, target, threshold, fg_extra_weight):
    """Weighted L1 loss for foreground pixels. Pixels with target > threshold are weighted by (1 + fg_extra_weight)."""
    weight = 1.0 + fg_extra_weight * (target > threshold).to(target.dtype)
    # sum over HW and channel, mean over batch
    num = (weight * torch.abs(pred - target)).sum()
    den = weight.sum()
    return num / (den + 1e-12)


def soft_dice_cd3_loss(pred, target, fg_threshold=0.05, eps=1e-8):
    """Soft Dice loss applied per-sample only where target has foreground.

    pred, target: [B,1,H,W]
    Returns batch-mean over valid samples; if none valid returns 0.0 tensor with grad.
    """
    pred = pred.float()
    target = target.float()

    B = pred.size(0)
    losses = []
    for i in range(B):
        p = pred[i].reshape(-1)
        t = target[i].reshape(-1)
        mask = (t > fg_threshold).float()
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            continue
        intersection = (p * mask).sum()
        dice = (2.0 * intersection + eps) / (p.sum() + mask_sum + eps)
        losses.append(1.0 - dice)

    if len(losses) == 0:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def mihc_recon_loss(pred, target, alpha=1.0, fg_threshold=0.05, fg_extra_weight=1.0, channel_weights=None,
                    pearson_weight=0.0, pearson_scales=None, pearson_scale_weights=None, cd3_dice_weight=0.0):
    """mIHC three-channel reconstruction loss. Computes per-channel losses and combines them with channel_weights.
    The per-channel loss is a combination of foreground-weighted L1 and SSIM, with optional differentiable Pearson correlation loss. 
    The CD3 Dice loss is applied to channel 1 when cd3_dice_weight > 0.0. The default parameters are set to maintain compatibility with previous
    """
    if channel_weights is None:
        channel_weights = [1.0, 1.0, 1.0]

    losses = []
    for c in range(3):
        p = pred[:, c:c+1]
        t = target[:, c:c+1]

        l1 = weighted_l1(p, t, fg_threshold, fg_extra_weight)
        ssim = ssim_loss(p, t)
        channel_loss = l1 + alpha * ssim

        # Only apply Pearson / multi-scale Pearson to HE->mIHC when pearson_weight > 0
        if pearson_weight > 0.0:
            if pearson_scales is None or pearson_scale_weights is None:
                channel_loss = channel_loss + pearson_weight * pearson_loss(p, t)
            else:
                channel_loss = channel_loss + pearson_weight * multi_scale_pearson_loss(p, t, pearson_scales, pearson_scale_weights)

        # Only apply CD3 Dice to channel index 1 when cd3_dice_weight > 0
        if c == 1 and cd3_dice_weight > 0.0:
            channel_loss = channel_loss + cd3_dice_weight * soft_dice_cd3_loss(p, t, fg_threshold)

        losses.append(channel_loss)

    weights = pred.new_tensor(channel_weights)
    total = sum(w * l for w, l in zip(weights, losses))
    # multiply by (3 / sum(weights))
    total = total * (3.0 / float(weights.sum()))
    return total


def st_recon_loss(pred, target):
    """ST reconstruction loss"""
    return F.mse_loss(pred, target)