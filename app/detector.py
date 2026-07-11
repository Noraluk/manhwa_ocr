"""Speech-bubble / narration-box detection and SFX filtering.

Manhwa dialogue and narration live inside light, enclosed containers
(white bubbles, light rectangular caption boxes). Onomatopoeia / SFX is
stylized lettering drawn directly over the artwork. We exploit that:

  1. Detect light enclosed regions (bubbles + caption boxes) via CV.
  2. A text line is kept only if it sits inside such a container OR its own
     local background is clean and light (covers boxes the mask misses).
  3. Everything else — text over busy/dark artwork — is treated as SFX
     and dropped.

This needs no model download and runs offline. For higher accuracy on
unusual art, swap in a trained detector (comic-text-detector / a YOLO
speech-bubble model) behind `detect_bubbles`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .geometry import Box, overlap_ratio

# --- tunables ---------------------------------------------------------------
LIGHT_THRESH = 200          # a pixel brighter than this counts as "paper"
MIN_BUBBLE_AREA = 1500      # px^2; ignore tiny light specks
CLOSE_KERNEL = 21           # morph close to seal text holes inside a bubble
RECT_FILL = 0.88            # contour/bbox fill above this => rectangular box
CLEAN_BG_MEAN = 165         # local background brightness to accept a text line
CLEAN_BG_STD = 55           # local background uniformity (lower = cleaner)


@dataclass
class Bubble:
    box: Box
    kind: str  # "dialogue" | "narration"


def detect_bubbles(bgr: np.ndarray) -> List[Bubble]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    light = cv2.threshold(gray, LIGHT_THRESH, 255, cv2.THRESH_BINARY)[1]

    # Seal the dark text strokes inside a bubble so it becomes one solid blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    solid = cv2.morphologyEx(light, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = bgr.shape[0] * bgr.shape[1]

    bubbles: List[Bubble] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h
        if rect_area < MIN_BUBBLE_AREA:
            continue
        if rect_area > 0.35 * img_area:  # whole-panel light region, not a bubble
            continue

        fill = cv2.contourArea(c) / rect_area if rect_area else 0.0
        if fill < 0.55:  # not a compact blob (stray strokes, borders)
            continue
        if not _contains_dark_text(gray, c):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        kind = "narration" if (fill >= RECT_FILL and len(approx) <= 6) else "dialogue"
        bubbles.append(Bubble(box=(x, y, w, h), kind=kind))

    return bubbles


def _contains_dark_text(gray: np.ndarray, contour: np.ndarray) -> bool:
    # Measure dark pixels strictly INSIDE the light blob (not its bbox corners,
    # which for a round bubble on dark art are background, not text).
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    interior = gray[mask > 0]
    if interior.size < 50:
        return False
    dark_frac = float(np.mean(interior < 100))
    # Real text fills a small-but-nonzero fraction; a plain white blob has ~none.
    return 0.005 < dark_frac < 0.6


def find_container(line_box: Box, bubbles: List[Bubble]) -> Optional[Bubble]:
    """Return the bubble that best encloses a text line, if any."""
    best, best_ov = None, 0.0
    for b in bubbles:
        ov = overlap_ratio(line_box, b.box)
        if ov > best_ov:
            best, best_ov = b, ov
    return best if best_ov >= 0.6 else None


def background_is_clean(gray: np.ndarray, box: Box) -> Tuple[bool, float]:
    """Analyse the non-text background right around a text line.

    Returns (is_clean, mean_brightness). Clean == light and uniform, i.e. the
    text sits on paper (bubble/caption) rather than over artwork.
    """
    x, y, w, h = box
    pad = max(3, int(0.15 * h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return False, 0.0

    # Otsu splits glyphs (dark) from background (light); measure the light class.
    thr, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg = roi[roi >= thr]
    if bg.size < 20:
        return False, 0.0
    bg_mean, bg_std = float(bg.mean()), float(bg.std())
    clean = bg_mean >= CLEAN_BG_MEAN and bg_std <= CLEAN_BG_STD
    return clean, bg_mean
