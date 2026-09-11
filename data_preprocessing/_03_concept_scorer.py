import cv2
import numpy as np


def calculate_otsu_thresholds(mihc_mega: np.ndarray) -> list:
    """
    Compute one Otsu threshold for each mIHC channel.

    Args:
        mihc_mega: Three-channel mIHC image with shape [3, H, W].
            Channel order: 0=DAPI, 1=CD3, 2=panCK.

    Returns:
        Three Otsu thresholds, one per channel.
    """
    thresholds = []
    for c in range(3):
        channel_matrix = mihc_mega[c]
        t_val, _ = cv2.threshold(
            channel_matrix,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        thresholds.append(t_val)

    return thresholds


def calculate_patch_scores(mihc_patch: np.ndarray, thresholds: list) -> list:
    """
    Compute the three continuous concept scores from an mIHC patch.

    For each channel, pixels above the corresponding Otsu threshold are
    considered positive. The concept score is the fraction of positive pixels
    over the full spatial field.

    Args:
        mihc_patch: Three-channel mIHC image with shape [3, H, W].
        thresholds: Three channel-specific thresholds.

    Returns:
        Three positive-pixel fractions in DAPI, CD3, panCK order.
    """
    scores = []
    total_pixels = mihc_patch.shape[1] * mihc_patch.shape[2]

    for c in range(3):
        channel_matrix = mihc_patch[c]
        t_val = thresholds[c]
        binary_mask = (channel_matrix > t_val).astype(np.float32)
        positive_count = np.sum(binary_mask)
        score = float(positive_count / total_pixels)
        scores.append(score)

    return scores