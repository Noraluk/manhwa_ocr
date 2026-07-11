# Manhwa OCR API

A small, self-hostable HTTP service that extracts **dialogue and narration** text
from webtoon / manhwa images and returns **where** each text block sits — so a
downstream tool can paint over it and drop in a translation. Onomatopoeia /
sound-effect lettering (SFX), series logos, and scanlation watermarks are
deliberately **skipped**.

Built with FastAPI + PaddleOCR + OpenCV. One streaming endpoint, no database, no
external API keys.

- **Two boxes per block** — a paint-over region and a tight glyph box.
- **SFX / logo / watermark filtering** — only real dialogue comes back.
- **Streaming API** — `POST /ocr` returns one JSON line per image as it finishes,
  so a caller can persist results incrementally over a long chapter.
- **Pluggable OCR backend** — PaddleOCR (default), EasyOCR, or Surya.
- **Offline** — models run locally; nothing is sent to a third party.

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [API](#api)
- [OCR backends](#ocr-backends)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Limitations](#limitations)

## How it works

Long webtoon strips are thousands of pixels tall, and OCR alone can't tell
dialogue from SFX. The pipeline separates the two structurally:

```
image → slice into overlapping tiles → OCR each tile → merge & de-dup
      → detect speech bubbles / caption boxes (CV)
      → keep text inside a bubble, or on a clean light background
      → drop SFX / logos / watermarks → group lines into blocks → sort top-to-bottom
```

- **Tiling** ([app/pipeline.py](app/pipeline.py)) — 1600px tiles with generous
  overlap so a bubble is never cut in half; boxes are offset back to full-image
  coordinates and de-duplicated by IoU.
- **Bubble / box detection** ([app/detector.py](app/detector.py)) — finds light,
  enclosed regions containing dark text. Rectangular → `narration`, rounded →
  `dialogue`.
- **SFX filtering** — a text line is kept if it's inside a detected bubble, **or**
  its local background is light and uniform (a caption on a white page the mask
  didn't close). Text over busy/dark artwork, or very low-confidence misreads of
  artwork, is dropped.
- **Non-speech filtering** — extra rules remove things you don't want translated:
  - **Oversized lettering** (`MAX_LINE_H`) — onomatopoeia / SFX and logo art use
    much larger glyphs than dialogue.
  - **No real words** (`MIN_LETTERS`) — episode numbers (`64`) and SFX that OCR
    misreads into digits/symbols (`77`, `@ot`).
  - **Watermarks** — a scanlation URL/domain stamped on the art (`ASURASCANS.COM`,
    `www.foo.net`) is dropped; real dialogue never contains a web domain. Add
    recurring watermarks or scene signage via `OCR_DROP_TEXT`.
  - **Title card** — large stylized text near the very top is detected as the
    series logo; it and the credit lines below are dropped. A mid-chapter image
    (no big text at the top) is left untouched.
  - **End-of-episode footer** — the same logic mirrored to the bottom: on a long
    strip, a series logo + credit line at the very bottom is dropped. Only runs on
    tall images (`FOOTER_MIN_IMG_H`), so short pages are never clipped.

Each block gives you **two boxes**:

- `bbox` — the full bubble / caption region → the area **safe to paint over**.
- `text_bbox` — a tight box around the glyphs → **where to place** the translation.

## Quick start

### Docker (recommended)

The image installs the Paddle-only dependency set and bakes the OCR models in, so
the first request doesn't wait on a download.

```bash
docker build -t manhwa-ocr .
docker run --rm -p 7860:7860 manhwa-ocr
# API on http://127.0.0.1:7860
```

### Local (Python 3.10)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # full set (lets you switch backends)
uvicorn app.main:app --reload            # API on http://127.0.0.1:8000
```

> `requirements.txt` is the full/dev set (PaddleOCR + EasyOCR + Surya).
> `requirements-slim.txt` is Paddle-only and is what the Docker image uses.

## API

### `POST /ocr`

Send one or more images (`multipart/form-data`, field `files`); get back a
**stream of NDJSON** (`application/x-ndjson`) — one JSON line per image, emitted
as each finishes.

```bash
curl -sN -F "files=@page1.webp" -F "files=@page2.webp" -F "files=@page3.webp" \
     http://127.0.0.1:7860/ocr
```

```
{"status":"started","total":3}
{"index":0,"filename":"page1.webp","status":"ok","result":{ … }}
{"index":1,"filename":"page2.webp","status":"error","error":"File is not a readable image."}
{"index":2,"filename":"page3.webp","status":"ok","result":{ … }}
```

- The first line opens the stream immediately (before the first, slow OCR) so a
  proxy doesn't idle-time-out the connection.
- Then one line per image **in input order**, each a self-contained result you can
  persist right away. A bad image fails only its own line.
- No server-side job state — nothing to lose if the host restarts.
- Up to 30 images per request (`MAX_FILES`), 25 MB each. On CPU a whole chapter is
  minutes of work, so prefer **small batches** (a few images per request) over one
  giant connection.

> **Reading the stream:** use a client that doesn't buffer the whole body — `curl -N`,
> a Go `json.Decoder`, or `requests(stream=True)`. Tools like Postman / Swagger UI
> wait for the full response, so they'll *look* like it returns all at once.

### `GET /health`

```json
{ "status": "ok", "backend": "paddleocr" }
```

Readiness check — ping it first to warm a cold/sleeping host before sending work.

### Result shape (`status: "ok"`)

```json
{
  "image_size": { "width": 800, "height": 14222 },
  "backend": "paddleocr",
  "blocks": [
    {
      "id": 0,
      "type": "dialogue",
      "text": "EVERYONE'S SAFE...",
      "bbox":      { "x": 244, "y": 870, "w": 172, "h": 62 },
      "text_bbox": { "x": 251, "y": 877, "w": 158, "h": 48 },
      "confidence": 0.48
    }
  ]
}
```

### Consuming the stream from Go

`json.Decoder` reads one object per loop, so each result is stored as it arrives:

```go
resp, _ := http.DefaultClient.Do(req) // req = multipart POST to /ocr with "files"
defer resp.Body.Close()

dec := json.NewDecoder(resp.Body) // stream — do not buffer the whole response
for {
    var item OcrItem
    if err := dec.Decode(&item); err == io.EOF {
        break
    } else if err != nil {
        return err
    }
    if item.Status == "ok" {
        store(item) // e.g. upsert into MongoDB
    }
}
```

## OCR backends

Pick with `OCR_BACKEND`. Speeds are for an 800×6829 strip on a laptop CPU.

| backend | quality | speed | notes |
| --- | --- | --- | --- |
| `paddleocr` *(default)* | **high** | ~24s | best on italic comic lettering (reads apostrophes / stylized glyphs). `PADDLE_MODELS=mobile` → ~7s but the smaller model drops some word spaces |
| `easyocr` | medium | **~10s** | GPU/MPS-accelerated; more character errors (`I'M`→`TM`, `CRY...`→`CR_`, dropped apostrophes) |
| `hybrid` | highest | slowest | Surya + EasyOCR; Surya reads what it can, EasyOCR fills the rest |
| `surya` | high | medium | correct case & punctuation; skips stylized shout-bubbles on dark art |

The default is **PaddleOCR** (medium models, text-orientation off): on slanted
hand-lettering it reads characters EasyOCR can't — e.g. `I'M SO ... COULD CRY...`
where EasyOCR emits `TM SO ... CR_`. It is CPU-only (no GPU build on Apple
Silicon). For throughput set `OCR_BACKEND=easyocr` (GPU-accelerated) or keep Paddle
with `PADDLE_MODELS=mobile`. `surya`/`hybrid` need the `surya-ocr` package **and** a
`llama-server` binary (`brew install llama.cpp`).

## Configuration

All via environment variables.

| Var | Default | Meaning |
| --- | --- | --- |
| `OCR_BACKEND` | `auto` | `paddleocr` \| `easyocr` \| `hybrid` \| `surya` \| `auto` (=`paddleocr`) |
| `PADDLE_MODELS` | `medium` | Paddle tier: `medium` (accurate, ~24s) or `mobile` (faster ~7s, drops some word spaces) |
| `OCR_LANGS` | `en` | comma-separated language codes for the backend |
| `TEXT_CASE` | `upper` | `upper` normalises to caps (comic lettering is all-caps); `keep` preserves raw OCR casing |
| `FIX_DIGIT_CONFUSIONS` | `on` | fix digit↔letter slips in letter-dominant words (`5O`→`SO`, `B0DY`→`BODY`); pure numbers, ordinals (`1ST`) and codes (`B747`) are left alone |
| `SPELLCHECK` | `on` | fix letter↔letter OCR slips against a dictionary (`GLYS`→`GUYS`). Only non-words with a single close fix change. Protect names with `OCR_KNOWN_WORDS` |
| `OCR_KNOWN_WORDS` | *(empty)* | comma-separated words the spell-checker must treat as correct (character/place names), e.g. `KAEL,DDAGAEBI` |
| `FIX_SEMICOLONS` | `on` | normalise misread separators: `;`→`,` and word-final `:`→`.` (time like `3:00` is kept) |
| `OCR_DROP_TEXT` | *(empty)* | comma-separated, case-insensitive substrings to drop as watermarks/signage on top of the automatic URL filter, e.g. `asurascans,room hof&soju` |

Detection/filtering thresholds live at the top of
[app/detector.py](app/detector.py) and [app/pipeline.py](app/pipeline.py).

## Deployment

It's a plain Docker container that listens on **7860** — deploy it on any host
that runs containers:

```bash
docker build -t manhwa-ocr .
docker run -d -p 7860:7860 manhwa-ocr
```

Notes for common targets:

- **Any VM / VPS** — build and run as above; put it behind a reverse proxy for
  TLS. Simplest way to keep it always-on.
- **Google Cloud Run** — serverless and scales to zero (good for a job that fires
  in bursts). Cloud Run injects `$PORT` (default 8080), so change the `CMD` to bind
  `$PORT` instead of the fixed 7860.
- **GPU** — PaddleOCR has no MPS build on Apple Silicon; on Linux with CUDA use a
  GPU base image + `paddlepaddle-gpu` for a large speedup.
- **Hugging Face Spaces (Docker SDK)** — works if the account's free CPU tier is
  available to you. A Space reads a YAML front-matter block from the top of a
  `README.md` (`sdk: docker`, `app_port: 7860`); add it before pushing to a Space.
  ZeroGPU is **not** usable here (Gradio-SDK only).

The service does OCR **inline** and serialises images on a single worker (models
are not thread-safe), so it processes one image at a time and streams each result.
For scale, run several replicas behind a queue.

## Testing

```bash
pip install pytest httpx
pytest -q
```

Tests use a fake OCR engine, so they run in under a second without loading a model.

**Debug overlay** — draw the detected boxes onto an image to eyeball what was kept
vs dropped:

```bash
python -m scripts.visualize page.webp overlay.png
```

## Project structure

```
app/
  main.py       FastAPI app — POST /ocr (NDJSON stream), GET /health
  pipeline.py   orchestration: tile → OCR → filter → group → clean text
  detector.py   speech-bubble / caption-box detection (OpenCV)
  ocr.py        pluggable OCR backends (paddleocr | easyocr | surya | hybrid)
  geometry.py   box math (IoU, union, dedup)
  draw.py       debug overlay rendering
  schemas.py    Pydantic response models
scripts/visualize.py   CLI: run extraction and save an annotated image
tests/          fast tests with a fake engine
Dockerfile               Paddle-only image, models baked in, serves on :7860
requirements.txt         full/dev set (all backends)
requirements-slim.txt    Paddle-only (used by the image)
```

## Limitations

- **Bubble boundaries** get imprecise when the page background is itself white
  (the bubble merges with the page); `bbox` then falls back to a padded text box.
  For pixel-accurate masks, plug a trained detector (comic-text-detector or a YOLO
  speech-bubble model) into `detect_bubbles`.
- **Contextual OCR errors** (`YOUR`↔`YOU'RE`, a missing word) can't be fixed by any
  backend or rule without a language model — a downstream LLM translator resolves
  those at translation time.
- **Scene signage** rendered into the artwork (a shop sign, a poster) can look
  exactly like dialogue to the heuristics; if one recurs, drop it with
  `OCR_DROP_TEXT`. A trained comic-text detector is the robust fix.
- **Speed** — the default backend is CPU-only; a long strip takes tens of seconds.
  Use `PADDLE_MODELS=mobile` or `OCR_BACKEND=easyocr` for throughput, or a GPU host.
- `MIN_CONF_LOOSE` trades recall vs. false positives from artwork — raise it if you
  see junk blocks, lower it if faint real text is dropped.
