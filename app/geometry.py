"""Small geometry helpers working on (x, y, w, h) tuples and quads."""
from __future__ import annotations

from typing import List, Sequence, Tuple

Box = Tuple[int, int, int, int]  # x, y, w, h
Quad = Sequence[Sequence[float]]  # 4 points [[x, y], ...]


def quad_to_box(quad: Quad) -> Box:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    return int(x0), int(y0), int(round(x1 - x0)), int(round(y1 - y0))


def area(b: Box) -> int:
    return max(0, b[2]) * max(0, b[3])


def intersection(a: Box, b: Box) -> Box:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    return (x0, y0, x1 - x0, y1 - y0)


def iou(a: Box, b: Box) -> float:
    inter = area(intersection(a, b))
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def overlap_ratio(inner: Box, outer: Box) -> float:
    """Fraction of `inner` covered by `outer`."""
    inter = area(intersection(inner, outer))
    return inter / area(inner) if area(inner) else 0.0


def union_box(boxes: Sequence[Box]) -> Box:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def shift(b: Box, dx: int, dy: int) -> Box:
    return (b[0] + dx, b[1] + dy, b[2], b[3])


def pad(b: Box, px: int, w_max: int, h_max: int) -> Box:
    x = max(0, b[0] - px)
    y = max(0, b[1] - px)
    x1 = min(w_max, b[0] + b[2] + px)
    y1 = min(h_max, b[1] + b[3] + px)
    return (x, y, x1 - x, y1 - y)


def dedup_boxes(items: List[dict], key: str = "box", thr: float = 0.5) -> List[dict]:
    """Greedy NMS-style dedup: drop later boxes that overlap a kept one."""
    kept: List[dict] = []
    for it in sorted(items, key=lambda d: area(d[key]), reverse=True):
        if all(iou(it[key], k[key]) < thr for k in kept):
            kept.append(it)
    return kept
