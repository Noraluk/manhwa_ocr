---
title: Manhwa OCR API
emoji: 🗯️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Manhwa OCR mini-API

Extract dialogue / narration text from webtoon (manhwa) images and return **where**
each text block sits, so a downstream tool can paint over it and drop in a translation.
Onomatopoeia / sound-effect lettering (SFX) is deliberately **skipped**.

## How it works

Long webtoon strips are thousands of pixels tall, and OCR alone can't tell dialogue
from SFX. The pipeline separates the two structurally:

```
image → slice into overlapping tiles → OCR each tile → merge & de-dup
      → detect speech bubbles / caption boxes (CV)
      → keep text inside a bubble, or on a clean light background
      → drop text over artwork (SFX) → group lines into blocks → sort top-to-bottom
```

- **Tiling** ([app/pipeline.py](app/pipeline.py)) — tiles of 1600px with 200px overlap so a
  bubble is never cut in half; boxes are offset back to full-image coordinates and
  de-duplicated by IoU.
- **Bubble / box detection** ([app/detector.py](app/detector.py)) — finds light, enclosed
  regions containing dark text. Rectangular → `narration`, rounded → `dialogue`.
- **SFX filtering** — a text line is kept if it's inside a detected bubble, **or** its
  local background is light and uniform (a caption on white page the mask didn't close).
  Text over busy/dark artwork, or very low-confidence misreads of artwork, are dropped.
- **Non-speech filtering** — three more rules remove things you don't want translated:
  - **Oversized lettering** (`MAX_LINE_H`) → onomatopoeia / SFX and logo art use much
    larger glyphs than dialogue.
  - **No real words** (`MIN_LETTERS`) → episode numbers ("64") and SFX that OCR
    misreads into digits/symbols ("77", "@ot").
  - **Watermarks** → a scanlation URL/domain stamped on the art ("ASURASCANS.COM",
    "www.foo.net") is dropped; real dialogue never contains a web domain. Add
    recurring watermarks or scene signage the CV can't tell from dialogue via
    `OCR_DROP_TEXT`.
  - **Title card** → large stylized text near the very top is detected as the series
    logo; it and the credit lines just below are dropped. On a mid-chapter image (no
    big text at the top) nothing is removed, so ordinary dialogue is never clipped.
  - **End-of-episode footer** → the same logic, mirrored to the bottom: on a long
    strip, a series logo + credit/watermark line crammed into the bottom band and
    cut off from the last panel by a big empty gap is dropped. Only runs on tall
    images (`FOOTER_MIN_IMG_H`); a short single-screen image is never treated as a
    chapter end, so text near its bottom is kept.
- **OCR backend** ([app/ocr.py](app/ocr.py)) — pluggable, pick with `OCR_BACKEND`:

  | backend | quality | speed (800×6829 strip) | notes |
  | --- | --- | --- | --- |
  | `paddleocr` *(default)* | **high** | ~24s | best on the italic comic lettering (reads apostrophes / stylized glyphs); CPU-only on Apple Silicon. `PADDLE_MODELS=mobile` → ~7s, EasyOCR-class speed, but the smaller model drops some word spaces |
  | `easyocr` | medium | **~10s** | GPU/MPS-accelerated, full recall, more character errors (`I'M`→`TM`, `CRY...`→`CR_`, drops apostrophes); no extra binary |
  | `hybrid` | highest | slowest | Surya + EasyOCR; Surya reads what it can, EasyOCR fills what Surya drops |
  | `surya` | high text accuracy | medium | correct case, punctuation; **skips** stylized shout-bubbles on dark art (labels them images) |

  The default is **PaddleOCR** (medium models, orientation off) — on this comic's
  slanted hand-lettering it reads characters EasyOCR can't (e.g. `I'M SO ... COULD
  CRY...` where EasyOCR emits `TM SO ... CR_`). It has **no GPU/MPS build on Apple
  Silicon**, so it runs on CPU (~2.4x slower than EasyOCR). When throughput matters
  more than accuracy, set `OCR_BACKEND=easyocr` (GPU-accelerated), or keep Paddle
  and set `PADDLE_MODELS=mobile` for EasyOCR-class speed at some spacing cost.
  `surya`/`hybrid` are highest quality but need the `surya-ocr` package **and** a
  `llama-server` binary (`brew install llama.cpp`).

## Each block gives you two boxes

- `bbox` — the full bubble / caption region → the area **safe to paint over**.
- `text_bbox` — a tight box around the glyphs → **where to place** the translated text.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API — `POST /ocr`

One endpoint. Send one or more images (`multipart/form-data`, field `files`);
get back a **stream of NDJSON** (`application/x-ndjson`) — one JSON line per
image, emitted as each finishes:

```bash
curl -sN -F "files=@01.webp" -F "files=@02.webp" -F "files=@03.webp" \
     http://127.0.0.1:8000/ocr
```
```
{"status":"started","total":3}
{"index":0,"filename":"01.webp","status":"ok","result":{ … }}
{"index":1,"filename":"02.webp","status":"error","error":"File is not a readable image."}
{"index":2,"filename":"03.webp","status":"ok","result":{ … }}
```

- The first line opens the stream immediately (before the first, slow OCR) so a
  proxy doesn't idle-time-out the connection.
- Then one line per image **in input order**, each a self-contained result you
  can persist right away. A bad image fails only its own line.
- No server-side job state (nothing to lose when the host sleeps).
- Up to 30 images per request (`MAX_FILES`), 25 MB each. On the free CPU tier a
  whole chapter is minutes of work — send it in **small batches** (a few images
  per request) rather than one giant connection.

`GET /health` → `{"status":"ok","backend":"paddleocr"}` (readiness check; ping it
first to wake a sleeping Space before sending work).

### Each `result` (status `ok`)

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
req, _ := http.NewRequest("POST", ocrURL+"/ocr", body) // body = multipart w/ "files"
req.Header.Set("Content-Type", writer.FormDataContentType())
resp, _ := http.DefaultClient.Do(req)
defer resp.Body.Close()

dec := json.NewDecoder(resp.Body) // stream, don't buffer the whole response
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

## Deploy (free) — Hugging Face Spaces

The repo ships a `Dockerfile` and `requirements-slim.txt` (Paddle-only, no
torch/easyocr/surya) so the image stays lean, plus the `sdk: docker` frontmatter
at the top of this file that a Space reads.

1. Create a new **Space** → SDK **Docker** → **Blank**.
2. Push this repo to it:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space>
   git push space main
   ```
3. It builds, bakes the Paddle models into the image, and serves on port `7860`.
   Your API is then at `https://<user>-<space>.hf.space` (e.g. `POST /ocr`).

Honest notes for the free CPU tier: ~24s/image (CPU-only, no GPU), and the Space
**sleeps when idle** — the first request after a sleep reloads the model (a few
seconds) before responding. The Space is **public** unless you upgrade. For lower
latency set `PADDLE_MODELS=mobile` (faster, drops some word spaces) as a Space
variable, or switch to a GPU host.

## Visualize / debug

Draws the boxes onto the image so you can eyeball what was kept vs dropped:

```bash
python -m scripts.visualize 2-fa638.webp overlay.png
```

## Test

```bash
pip install pytest httpx
pytest -q
```

API tests use a fake OCR engine, so they run in <1s without loading a model.

## Configuration (env vars)

| Var           | Default   | Meaning                                        |
| ------------- | --------- | ---------------------------------------------- |
| `OCR_BACKEND` | `auto`    | `paddleocr` \| `easyocr` \| `hybrid` \| `surya` \| `auto` (=`paddleocr`; see table above) |
| `PADDLE_MODELS` | `medium` | Paddle model tier: `medium` (accurate, ~24s) or `mobile` (EasyOCR-class speed, drops some word spaces) |
| `OCR_LANGS`   | `en`      | comma-separated language codes for the backend |
| `TEXT_CASE`   | `upper`   | `upper` normalises output to caps (comic lettering is all-caps and OCR emits mixed case); `keep` preserves raw OCR casing |
| `FIX_DIGIT_CONFUSIONS` | `on` | fix OCR digit↔letter slips inside letter-dominant words (`5O`→`SO`, `B0DY`→`BODY`); pure numbers, ordinals (`1ST`), and digit-heavy codes (`B747`) are left alone. `off` to disable |
| `SPELLCHECK`   | `on` | fix letter↔letter OCR slips against a dictionary (`GLYS`→`GUYS`, `INJLRIES`→`INJURIES`, `WERENT`→`WEREN'T`). Only non-words with a single close fix are changed; real words are kept. **Caveat:** a name one edit from a common word (`KAEL`→`KEEL`) can be changed — protect names with `OCR_KNOWN_WORDS`, or set `off` |
| `OCR_KNOWN_WORDS` | *(empty)* | comma-separated words the spell-checker must treat as correct (character/place names), e.g. `KAEL,DDAGAEBI` |
| `FIX_SEMICOLONS` | `on` | normalise misread separators: `;`→`,` and a word-final `:`→`.` (comic lettering rarely uses either; time like `3:00` is kept). `off` if your text uses real `;`/`:` |
| `OCR_DROP_TEXT` | *(empty)* | comma-separated, case-insensitive substrings to drop as watermarks/signage on top of the automatic URL filter, e.g. `asurascans,room hof&soju`. A block is dropped if its text contains any of them |

Tunables for detection/filtering live at the top of
[app/detector.py](app/detector.py) and [app/pipeline.py](app/pipeline.py).

## Limitations & honest notes

- The CV bubble detector is a solid offline default, but bubble **boundaries** get
  imprecise when the page background is itself white (bubble merges with the page); the
  clean-background heuristic + a confidence floor cover recall in that case, and `bbox`
  falls back to a padded text box. For pixel-accurate bubble masks, plug a trained
  detector (comic-text-detector or a YOLO speech-bubble model) into `detect_bubbles`.
- OCR accuracy depends on the backend (see table above). The default `paddleocr`
  reads this comic's italic lettering best; `easyocr` is faster (GPU/MPS) but yields
  the odd `0`/`O`, `;`/`,` slip, dropped apostrophes, and stylized-glyph misreads
  (`CRY...`→`CR_`); `hybrid`/`surya` are highest quality but run heavy models.
  Some errors are contextual (`YOUR`↔`YOU'RE`, a missing word) and no backend or
  rule can fix them without a language model — a downstream LLM translator resolves
  those at translation time.
- No backend reads stylized SFX reliably (by design we drop those anyway); the
  series-logo title card (top) and end-of-episode footer (bottom) are removed
  heuristically — a trained comic-text detector is the robust path if either
  becomes important.
- `MIN_CONF_LOOSE` trades recall vs. false positives from artwork. Raise it if you see
  junk blocks; lower it if faint real text is dropped.
- PaddleOCR is **CPU-only on Apple Silicon** (no MPS build), so the default is
  ~2.4x slower than `easyocr`; a long strip takes tens of seconds. For real traffic,
  run extraction behind a queue; `/ocr` already serialises images on one worker
  and streams each result, so a caller can persist as it goes.
```
