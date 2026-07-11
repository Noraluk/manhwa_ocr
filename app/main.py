"""FastAPI OCR service for manhwa/webtoon text extraction.

One working endpoint, POST /ocr: send one or more images, get back an NDJSON
stream — one JSON line per image as its OCR finishes. Designed for a server-side
caller (e.g. a Go worker looping a chapter) that persists each result as it
arrives: the stream keeps bytes flowing during the minutes-long job (so proxies
don't time it out), a failed image only fails its own line, and there is no
server-side job state to lose when the host sleeps.
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .ocr import get_engine
from .pipeline import extract
from .schemas import OcrItem


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the OCR model once so the first request isn't slow.
    get_engine()
    yield


app = FastAPI(
    title="Manhwa OCR API",
    version="2.0.0",
    description="Extract dialogue/narration text and their bounding boxes from "
    "webtoon images (skipping SFX). POST /ocr streams one result per image.",
    lifespan=lifespan,
)

MAX_BYTES = 25 * 1024 * 1024      # per image
# Keep a request modest on the free CPU tier (~seconds/image): the caller should
# send a chapter in small batches rather than one giant connection.
MAX_FILES = 30


def _load_bgr(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="File is not a readable image.")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _process_one(index: int, filename: str, data: bytes) -> OcrItem:
    """OCR a single image; errors are isolated to this item so one bad image
    never aborts the rest of the batch."""
    try:
        if not data:
            raise HTTPException(status_code=422, detail="Empty file.")
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="Image too large (>25MB).")
        result = extract(_load_bgr(data))
        return OcrItem(index=index, filename=filename, status="ok", result=result)
    except HTTPException as e:
        return OcrItem(index=index, filename=filename, status="error", error=str(e.detail))
    except Exception as e:  # a bad image shouldn't abort the whole chapter
        return OcrItem(index=index, filename=filename, status="error", error=str(e))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": get_engine().name}


@app.post("/ocr")
async def ocr(files: List[UploadFile] = File(...)) -> StreamingResponse:
    """OCR one or more images, streaming NDJSON (`application/x-ndjson`).

    First line is `{"status":"started","total":N}` (sent immediately so the
    connection opens before the first, slow OCR). Then one `OcrItem` line per
    image, in input order, emitted as each finishes.
    """
    if not files:
        raise HTTPException(status_code=422, detail="No files provided.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=413, detail=f"Too many files (>{MAX_FILES}).")

    # Read raw bytes up front (small); decode+OCR one at a time while streaming so
    # we never hold every decoded strip (tens of MB each) in memory at once.
    payload = [((f.filename or f"image_{i}"), await f.read()) for i, f in enumerate(files)]

    async def stream() -> AsyncIterator[bytes]:
        yield ('{"status":"started","total":%d}\n' % len(payload)).encode()
        for i, (name, data) in enumerate(payload):
            item = await run_in_threadpool(_process_one, i, name, data)
            yield (item.model_dump_json() + "\n").encode()

    return StreamingResponse(stream(), media_type="application/x-ndjson")
