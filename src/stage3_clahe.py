"""Candidate Stage 3 preprocessing for the detector input (uint8 OpenCV BGR)."""

import math

import cv2
import numpy as np


class DetectorCLAHE:
    """Apply CLAHE to Lab L and return a BGR frame with the original geometry."""

    def __init__(self, clip_limit: float, tile_grid_size: tuple[int, int]):
        if not math.isfinite(clip_limit) or clip_limit <= 0:
            raise ValueError("clip_limit must be finite and positive")
        if len(tile_grid_size) != 2 or any(
            type(count) is not int or count <= 0 for count in tile_grid_size
        ):
            raise ValueError("tile_grid_size must contain two positive integers")
        self.equalizer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or not frame_bgr.size):
            raise ValueError("Expected a non-empty uint8 BGR frame of shape H×W×3")
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = self.equalizer.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
