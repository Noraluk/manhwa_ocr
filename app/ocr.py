"""Pluggable OCR backend.

Exposes a single `get_engine()` returning an object with:
    detect(bgr: np.ndarray) -> List[TextLine]

Backends:
  - paddleocr (default; best character accuracy here, CPU-only on Apple Silicon)
  - easyocr   (fast, GPU/MPS via torch; set OCR_BACKEND=easyocr — more char errors)
  - surya / hybrid (transformer OCR; need llama.cpp — see _Surya)

Selection order: env OCR_BACKEND, else the auto default (PaddleOCR).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import numpy as np

from .geometry import Quad, iou, quad_to_box


@dataclass
class TextLine:
    text: str
    quad: Quad          # 4 points in the coords of the image passed to detect()
    confidence: float

    @property
    def box(self):
        return quad_to_box(self.quad)


class _EasyOCR:
    name = "easyocr"

    def __init__(self, langs: Optional[List[str]] = None):
        import easyocr  # lazy
        import torch

        # Use the GPU when present (CUDA, or Apple-Silicon Metal/MPS) — big speedup.
        gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        self._reader = easyocr.Reader(langs or ["en"], gpu=gpu)

    def detect(self, bgr: np.ndarray) -> List[TextLine]:
        # easyocr expects RGB; it also accepts a numpy array directly.
        rgb = bgr[:, :, ::-1]
        out = []
        for quad, text, conf in self._reader.readtext(rgb, detail=1, paragraph=False):
            text = (text or "").strip()
            if text:
                out.append(TextLine(text=text, quad=quad, confidence=float(conf)))
        return out


class _PaddleOCR:
    """PaddleOCR 3.x. Best character accuracy on the comic's italic hand-lettering
    here: reads apostrophes and stylized glyphs EasyOCR misses (e.g. "I'M SO ...
    COULD CRY..." where EasyOCR emits "TM SO ... CR_"). CPU-only on Apple Silicon
    (no MPS), so ~2.4x slower than EasyOCR with the default (medium) models.

    Orientation and doc-unwarping are OFF on purpose: comic lettering is neither
    rotated nor a scanned document page, and the textline-orientation classifier
    actively garbles stylized text. Set PADDLE_MODELS=mobile for the lightweight
    models (EasyOCR-class speed, but the smaller recogniser drops some word
    spaces, e.g. "IN DANGER" -> "INDANGER").
    """

    name = "paddleocr"

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR  # lazy

        kw = dict(
            lang=lang,
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        if os.getenv("PADDLE_MODELS", "medium").lower() == "mobile":
            kw["text_detection_model_name"] = "PP-OCRv5_mobile_det"
            kw["text_recognition_model_name"] = "PP-OCRv5_mobile_rec"
        self._ocr = PaddleOCR(**kw)

    def detect(self, bgr: np.ndarray) -> List[TextLine]:
        out: List[TextLine] = []
        for res in self._ocr.predict(input=bgr) or []:
            d = res if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
            texts = d.get("rec_texts", [])
            scores = d.get("rec_scores", [])
            polys = d.get("rec_polys", d.get("dt_polys", []))
            for text, conf, poly in zip(texts, scores, polys):
                text = (text or "").strip()
                if not text:
                    continue
                quad = [[float(p[0]), float(p[1])] for p in poly]
                out.append(TextLine(text=text, quad=quad, confidence=float(conf)))
        return out


class _Surya:
    """State-of-the-art open-source OCR (transformer model served via llama.cpp,
    Metal-accelerated on Apple Silicon). Best recognition quality here: correct
    case, keeps apostrophes, and stays silent on stylized SFX instead of
    hallucinating. Returns per-block text; we split it into per-line boxes so the
    rest of the pipeline (height filter, clustering) works unchanged.

    Requires the `surya-ocr` package and a `llama-server` binary
    (`brew install llama.cpp`, or set LLAMA_CPP_BINARY).
    """

    name = "surya"
    _TAG = re.compile(r"<[^>]+>")
    _BR = re.compile(r"</p>\s*<p>|<br\s*/?>", re.I)

    def __init__(self):
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        self._rec = RecognitionPredictor(SuryaInferenceManager())

    def _lines_from_html(self, html: str) -> List[str]:
        if not html:
            return []
        parts = self._BR.split(html)
        return [t for t in (self._TAG.sub("", p).strip() for p in parts) if t]

    def detect(self, bgr: np.ndarray) -> List[TextLine]:
        from PIL import Image

        pil = Image.fromarray(bgr[:, :, ::-1])  # BGR -> RGB
        page = self._rec([pil], full_page=True)[0]
        out: List[TextLine] = []
        for blk in page.blocks:
            lines = self._lines_from_html(blk.html or "")
            if not lines:
                continue
            xs = [p[0] for p in blk.polygon]
            ys = [p[1] for p in blk.polygon]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            conf = max(0.0, min(1.0, float(blk.confidence)))
            band = (y1 - y0) / len(lines)  # slice the block into per-line rows
            for i, text in enumerate(lines):
                ly0, ly1 = y0 + i * band, y0 + (i + 1) * band
                quad = [[x0, ly0], [x1, ly0], [x1, ly1], [x0, ly1]]
                out.append(TextLine(text=text, quad=quad, confidence=conf))
        return out


class _Hybrid:
    """Best overall quality: Surya reads what it can (accurate case, spelling,
    punctuation) and EasyOCR fills the regions Surya declines — chiefly stylized
    shout-bubbles on dark art, which Surya labels as images. We keep every Surya
    line and add only EasyOCR lines that don't overlap one, so Surya wins on
    shared regions and EasyOCR supplies the recall Surya lacks. Slower (two
    models per tile), but loses the least text.
    """

    name = "hybrid"
    _OVERLAP_THR = 0.2

    def __init__(self, langs: Optional[List[str]] = None):
        self._surya = _Surya()
        self._easy = _EasyOCR(langs)

    def detect(self, bgr: np.ndarray) -> List[TextLine]:
        surya_lines = self._surya.detect(bgr)
        merged = list(surya_lines)
        for el in self._easy.detect(bgr):
            if all(iou(el.box, sl.box) < self._OVERLAP_THR for sl in surya_lines):
                merged.append(el)
        return merged


@lru_cache(maxsize=1)
def get_engine():
    backend = os.getenv("OCR_BACKEND", "auto").lower()
    langs = [s.strip() for s in os.getenv("OCR_LANGS", "en").split(",") if s.strip()]

    if backend == "hybrid":
        return _Hybrid(langs)
    if backend == "surya":
        return _Surya()
    if backend == "easyocr":
        return _EasyOCR(langs)
    if backend == "paddleocr":
        return _PaddleOCR(langs[0] if langs else "en")

    # auto: PaddleOCR (medium models, orientation off). Best character accuracy on
    # this comic's italic lettering — reads apostrophes / stylized glyphs EasyOCR
    # drops. ~2.4x slower (CPU-only on Apple Silicon). For EasyOCR-class speed set
    # OCR_BACKEND=easyocr, or PADDLE_MODELS=mobile to keep Paddle but go faster.
    return _PaddleOCR(langs[0] if langs else "en")


def _surya_available() -> bool:
    import shutil

    try:
        import surya  # noqa: F401
    except Exception:
        return False
    return bool(os.getenv("LLAMA_CPP_BINARY") or shutil.which("llama-server"))
