"""Image loading, preprocessing and HOG feature extraction for baseline1."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.feature import hog


SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def load_image_bgr(path: str) -> np.ndarray:
    """Load an image as a BGR uint8 array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If OpenCV cannot decode the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    # Read via bytes + imdecode: cv2.imread cannot handle non-ASCII (e.g. Chinese)
    # paths on Windows, whereas fromfile/imdecode decodes the byte buffer directly.
    buf = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def preprocess(bgr: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Convert to grayscale and resize to (width, height) = image_size."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    width, height = image_size
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def extract_hog(bgr: np.ndarray, image_size: Tuple[int, int],
                hog_params: Dict) -> np.ndarray:
    """Extract the HOG feature vector from a BGR image.

    Parameters
    ----------
    bgr : BGR image.
    image_size : (width, height) the grayscale image is resized to.
    hog_params : dict with orientations, pixels_per_cell, cells_per_block,
        block_norm, transform_sqrt, channel_axis.
    """
    gray = preprocess(bgr, image_size)
    pixels_per_cell = tuple(hog_params.get("pixels_per_cell", [16, 16]))
    cells_per_block = tuple(hog_params.get("cells_per_block", [2, 2]))
    features = hog(
        gray,
        orientations=hog_params.get("orientations", 9),
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=hog_params.get("block_norm", "L2-Hys"),
        transform_sqrt=hog_params.get("transform_sqrt", True),
        feature_vector=True,
        channel_axis=hog_params.get("channel_axis", None),
    )
    return features.astype(np.float32)


def hog_vector_length(image_size: Tuple[int, int], hog_params: Dict) -> int:
    """Return the expected HOG vector length (for shape sanity checks)."""
    width, height = image_size
    ppc_h, ppc_w = hog_params.get("pixels_per_cell", [16, 16])
    cpb_h, cpb_w = hog_params.get("cells_per_block", [2, 2])
    cells_y = height // ppc_h
    cells_x = width // ppc_w
    blocks_y = cells_y - cpb_h + 1
    blocks_x = cells_x - cpb_w + 1
    orientations = hog_params.get("orientations", 9)
    return blocks_y * blocks_x * orientations * cpb_h * cpb_w
