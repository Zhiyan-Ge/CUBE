import os
import re
import cv2
import numpy as np

WSI_WIDTH_BASE = 22541
WSI_HEIGHT_BASE = 64216


def extract_and_normalize_coordinates(filename: str) -> list:
    """
    Extract the large-patch coordinates from the filename and normalize them
    relative to the whole-slide image dimensions.
    """
    match = re.search(r'\[(\d+)\s*,\s*(\d+)\]', filename)
    if match:
        x_raw = float(match.group(1))
        y_raw = float(match.group(2))
    else:
        numbers = re.findall(r'\d+', filename)
        x_raw, y_raw = float(numbers[0]), float(numbers[1])

    return [x_raw / WSI_WIDTH_BASE, y_raw / WSI_HEIGHT_BASE]


def extract_patch_grid_coord(filename: str) -> list:
    """
    Extract the patch_i_j grid indices from a HEMIT filename.

    Example:
        [6407,49798]_patch_0_2.tif -> [0, 2]

    The original dataset indices are stored as provided and are not assumed
    to represent physical pixel offsets.
    """
    match = re.search(r'_patch_(\d+)_(\d+)', filename)
    if not match:
        raise ValueError(f"Cannot parse patch_grid_coord from filename: {filename}")

    patch_i = int(match.group(1))
    patch_j = int(match.group(2))
    return [patch_i, patch_j]


def read_he_image(file_path: str) -> np.ndarray:
    """
    Read an H&E image and return a [3, 1024, 1024] RGB array.
    """
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.transpose(2, 0, 1)


def read_mihc_image(file_path: str) -> np.ndarray:
    """
    Read an mIHC image without channel reordering beyond BGR-to-RGB conversion,
    preserving the physical RGB channel order as [3, 1024, 1024].
    """
    img = cv2.imread(file_path)  # OpenCV reads images as [H, W, C] in BGR order.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.transpose(2, 0, 1)  # Convert [H, W, C] to [C, H, W].


def process_single_paired_patch(he_path: str, mihc_path: str) -> dict:
    """
    Read one paired H&E/mIHC patch together with its spatial coordinates.
    """
    filename = os.path.basename(he_path)
    normalized_mega_coord = extract_and_normalize_coordinates(filename)
    patch_grid_coord = extract_patch_grid_coord(filename)
    he_matrix = read_he_image(he_path)
    mihc_matrix = read_mihc_image(mihc_path)

    return {
        "normalized_mega_coord": normalized_mega_coord,
        "patch_grid_coord": patch_grid_coord,
        "he_matrix_1024": he_matrix,
        "mihc_matrix_1024": mihc_matrix,
    }