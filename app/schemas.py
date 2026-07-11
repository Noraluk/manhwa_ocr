"""Pydantic response models for the OCR API."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    dialogue = "dialogue"
    narration = "narration"


class Size(BaseModel):
    width: int
    height: int


class BBox(BaseModel):
    """Axis-aligned box in original-image pixel coordinates."""

    x: int
    y: int
    w: int
    h: int


class Block(BaseModel):
    id: int
    type: BlockType
    text: str
    # Full bubble / narration-box region — the area safe to paint over.
    bbox: BBox
    # Tight box around the actual glyphs — where translated text should sit.
    text_bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractResponse(BaseModel):
    image_size: Size
    backend: str
    blocks: List[Block]


class OcrItem(BaseModel):
    """One line of the /ocr NDJSON stream: the result for a single input image."""

    index: int                       # position of this image in the request
    filename: str
    status: str                      # "ok" | "error"
    result: Optional[ExtractResponse] = None   # present when status == "ok"
    error: Optional[str] = None                # present when status == "error"
