"""Run the pipeline on an image, print JSON, and save an annotated overlay.

Usage: python -m scripts.visualize <image> [out.png]
"""
import json
import sys

import cv2

from app.draw import draw_overlay
from app.pipeline import extract


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "2-fa638.webp"
    out = sys.argv[2] if len(sys.argv) > 2 else "overlay.png"

    bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    if bgr is None:  # webp via cv2 can be flaky; fall back to PIL
        import numpy as np
        from PIL import Image
        bgr = cv2.cvtColor(np.array(Image.open(src).convert("RGB")), cv2.COLOR_RGB2BGR)

    resp = extract(bgr)
    print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))

    cv2.imwrite(out, draw_overlay(bgr, resp.blocks))
    print(f"\n{len(resp.blocks)} blocks | backend={resp.backend} | overlay -> {out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
