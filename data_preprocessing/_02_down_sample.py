#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUBE image resizing utilities.

This module resizes paired 1024x1024 H&E and mIHC patches to the image sizes
used by CUBE. It does not handle coordinates, patch extraction, or file I/O.

Shape convention:
    input:  [3, H, W]
    output: [3, target_size, target_size]
"""

from typing import Tuple
import cv2
import numpy as np


DEFAULT_TARGET_SIZE = 256


def _validate_chw_image(image: np.ndarray, name: str = "image") -> None:
    """
    Validate that the input is a three-channel CHW NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(image)}")

    if image.ndim != 3:
        raise ValueError(f"{name} must have shape [C, H, W], got {image.shape}")

    if image.shape[0] != 3:
        raise ValueError(f"{name} must have 3 channels, got {image.shape}")

    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError(f"{name} has invalid spatial dimensions: {image.shape}")


def downsample_chw_image(
    image: np.ndarray,
    target_size: int = DEFAULT_TARGET_SIZE,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize a single CHW image.

    Args:
        image: Image array with shape [3, H, W], typically [3, 1024, 1024].
        target_size: Target height and width.
        interpolation: OpenCV interpolation mode. INTER_AREA is used by default.

    Returns:
        Resized image with shape [3, target_size, target_size].
    """
    _validate_chw_image(image, name="image")

    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError(f"target_size must be a positive integer, got {target_size}")

    if image.shape[1:] == (target_size, target_size):
        return image.copy()

    image_hwc = np.transpose(image, (1, 2, 0))
    resized_hwc = cv2.resize(
        image_hwc,
        (target_size, target_size),
        interpolation=interpolation,
    )
    resized_chw = np.transpose(resized_hwc, (2, 0, 1))
    return np.ascontiguousarray(resized_chw)


def downsample_paired_images(
    he_mega: np.ndarray,
    mihc_mega: np.ndarray,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize paired H&E and mIHC images to the same spatial resolution.

    Args:
        he_mega: H&E array with shape [3, H, W].
        mihc_mega: mIHC array with shape [3, H, W].
        target_size: Target height and width.

    Returns:
        he_matrix: Resized H&E array.
        mihc_matrix: Resized mIHC array.
    """
    _validate_chw_image(he_mega, name="he_mega")
    _validate_chw_image(mihc_mega, name="mihc_mega")

    he_matrix = downsample_chw_image(
        image=he_mega,
        target_size=target_size,
        interpolation=cv2.INTER_AREA,
    )
    mihc_matrix = downsample_chw_image(
        image=mihc_mega,
        target_size=target_size,
        interpolation=cv2.INTER_AREA,
    )

    return he_matrix, mihc_matrix


def prepare_cube_images(he_1024, mihc_1024):
    """
    Prepare the image tensors stored for CUBE.

    H&E is stored at both 512x512 and 256x256. mIHC is stored at 512x512.
    All outputs are scaled to [0, 1] and saved as float16.
    """
    he_512 = downsample_chw_image(he_1024, 512)
    mihc_512 = downsample_chw_image(mihc_1024, 512)
    he_256 = downsample_chw_image(he_1024, 256)

    he_512 = (he_512.astype(np.float32) / 255.0).astype(np.float16)
    mihc_512 = (mihc_512.astype(np.float32) / 255.0).astype(np.float16)
    he_256 = (he_256.astype(np.float32) / 255.0).astype(np.float16)

    return he_512, mihc_512, he_256