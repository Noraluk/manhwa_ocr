"""Draw extraction results onto an image for visual debugging."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .schemas import Block

# BGR colours per block type
COLORS = {"dialogue": (0, 200, 0), "narration": (255, 140, 0)}
TEXT_BOX_COLOR = (0, 0, 255)


def draw_overlay(bgr: np.ndarray, blocks: List[Block]) -> np.ndarray:
    """Return a copy of `bgr` with each block's paint box (green/orange),
    text box (red), and id/type label drawn on it."""
    canvas = bgr.copy()
    for b in blocks:
        c = COLORS.get(b.type.value, (0, 200, 0))
        cv2.rectangle(canvas, (b.bbox.x, b.bbox.y),
                      (b.bbox.x + b.bbox.w, b.bbox.y + b.bbox.h), c, 3)
        cv2.rectangle(canvas, (b.text_bbox.x, b.text_bbox.y),
                      (b.text_bbox.x + b.text_bbox.w, b.text_bbox.y + b.text_bbox.h),
                      TEXT_BOX_COLOR, 1)
        cv2.putText(canvas, f"{b.id}:{b.type.value}", (b.bbox.x, max(12, b.bbox.y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return canvas
