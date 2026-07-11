"""End-to-end extraction: slice -> OCR -> detect bubbles -> filter -> group."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from . import geometry as geo
from .detector import Bubble, background_is_clean, detect_bubbles, find_container
from .ocr import TextLine, get_engine
from .schemas import BBox, Block, BlockType, ExtractResponse, Size

TILE_H = 1600
# Overlap must exceed the tallest bubble so every bubble fits wholly inside the
# one tile that "owns" it (see _ocr_all_tiles). A bubble clipped at a tile edge
# is read badly by every backend — EasyOCR garbles it, block-OCR backends drop
# it — so keep this generous. It costs little time (MPS/GPU dominates, not tile
# count), so don't shrink it for speed.
OVERLAP = 600
DEDUP_IOU = 0.5
# Text inside a detected bubble is trusted even at low OCR confidence. Text on a
# merely-light background lacks that validation, so it must clear this floor —
# it screens out artwork (claws, highlights) that OCR misreads as a few glyphs.
MIN_CONF_LOOSE = 0.15
# A detected bubble taller than this multiple of its text cluster is treated as
# a merge of several bubbles; its box is discarded in favour of a per-cluster box.
BUBBLE_FIT_RATIO = 2.0
# Dialogue/narration glyphs are small (~20-80px tall here). SFX and logo art use
# much larger lettering, so an OCR line taller than this is dropped as non-speech.
MAX_LINE_H = 120
# A block must contain at least this many alphabetic characters. Kills episode
# numbers ("64") and SFX that OCR misreads into digits/symbols ("77", "@ot").
MIN_LETTERS = 2
# Title-card auto-detection. Two signals, whichever reaches lower:
#   A) large stylized logo text near the very top (works when OCR reads the logo)
#   B) an isolated text group at the very top followed by a big empty gap before
#      the story starts (catches the credit line even when the logo isn't OCR'd)
# Neither fires on a mid-chapter image, so ordinary dialogue is never clipped.
TITLE_SCAN_PX = 1000
LOGO_MIN_H = 120
TITLE_MARGIN = 250
TITLE_GAP = 1500
# End-of-episode footer (series logo + credit/watermark line) sits at the very
# bottom of a chapter's last image. It's anchored on the oversized series logo in
# this bottom band; from the logo down is footer. See _footer_band_top for why we
# don't use a gap heuristic here (it would clip real dialogue that sits low).
FOOTER_SCAN_PX = 1000
# A footer only appears on a chapter's last image, which is always a long strip.
# Below this height the image is a single screen (or a test fixture), never an
# end-of-episode page, so footer detection stays off and can't clip real dialogue.
FOOTER_MIN_IMG_H = 3000
# Comic lettering is essentially always uppercase, but OCR of stylized fonts
# emits mixed case ("ThinK", "Going"). Normalise so output matches the art.
# Set TEXT_CASE=keep to preserve the raw OCR casing.
TEXT_CASE = os.getenv("TEXT_CASE", "upper").lower()


def _letter_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _normalize_case(text: str) -> str:
    return text.upper() if TEXT_CASE == "upper" else text


# Common OCR glyph confusions: a digit inside a mostly-letter word is almost
# always a misread letter in comic lettering (e.g. "5O" -> "SO", "B0DY" ->
# "BODY"). Off via FIX_DIGIT_CONFUSIONS=off if your text has real alphanumerics.
_DIGIT_TO_LETTER = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})
_ORDINAL = re.compile(r"^\d+(ST|ND|RD|TH)$", re.I)
FIX_DIGITS = os.getenv("FIX_DIGIT_CONFUSIONS", "on").lower() != "off"


# A quote/backtick between two letters is a misread apostrophe (IT"S -> IT'S).
_MIDWORD_QUOTE = re.compile(r'(?<=[A-Za-z])["“”`´](?=[A-Za-z])')
# A colon right after a word (not between digits like "3:00") is a misread
# sentence dot ("DEATH ITSELF:" -> "DEATH ITSELF.").
_WORD_COLON = re.compile(r"(?<=[A-Za-z]):(?=\s|$)")
# Comic lettering rarely uses ";" or sentence ":"; OCR reads "," / "." as these.
FIX_SEMICOLONS = os.getenv("FIX_SEMICOLONS", "on").lower() != "off"


def _fix_punctuation(text: str) -> str:
    text = _MIDWORD_QUOTE.sub("'", text)
    if FIX_SEMICOLONS:
        text = text.replace(";", ",")
        text = _WORD_COLON.sub(".", text)
    return text


def _fix_digit_confusions(text: str) -> str:
    if not FIX_DIGITS:
        return text
    fixed = []
    for tok in text.split(" "):
        letters = sum(c.isalpha() for c in tok)
        digits = sum(c.isdigit() for c in tok)
        # Fix only a letter-dominant mix (leaves pure numbers, ordinals like
        # "1ST", and digit-heavy codes like "B747" — likely real — untouched).
        if letters and digits and letters >= digits and not _ORDINAL.match(tok):
            fixed.append(tok.translate(_DIGIT_TO_LETTER))
        else:
            fixed.append(tok)
    return " ".join(fixed)


# Spell-fix: correct OCR letter↔letter slips ("GLYS"->"GUYS") against a
# dictionary. Only NON-words with a single edit-distance-1 fix are touched, so
# real words and unusual names (no close match) are left alone — but a name that
# happens to sit one edit from a common word (e.g. "KAEL"->"KEEL") can still be
# changed. Protect such names via OCR_KNOWN_WORDS, or disable with SPELLCHECK=off.
SPELLCHECK = os.getenv("SPELLCHECK", "on").lower() != "off"
_TOKEN = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")  # one alpha run only
_speller = None


def _get_speller():
    global _speller
    if _speller is None:
        from spellchecker import SpellChecker

        sp = SpellChecker(distance=1)
        extra = [w.strip() for w in os.getenv("OCR_KNOWN_WORDS", "").split(",") if w.strip()]
        if extra:
            sp.word_frequency.load_words([w.lower() for w in extra])
        _speller = sp
    return _speller


def _spellfix(text: str) -> str:
    if not SPELLCHECK:
        return text
    sp = _get_speller()
    out = []
    for tok in text.split(" "):
        m = _TOKEN.match(tok)  # skip compounds (hyphen/apostrophe) and digit tokens
        if not m:
            out.append(tok)
            continue
        pre, word, post = m.groups()
        lw = word.lower()
        if len(word) < 4 or lw in sp:  # too short to trust, or already a word
            out.append(tok)
            continue
        corr = sp.correction(lw)
        if corr and corr != lw and corr in sp:
            corr = corr.upper() if word.isupper() else corr
            out.append(pre + corr + post)
        else:
            out.append(tok)
    return " ".join(out)


# Scanlation watermarks are a URL/domain stamped on the artwork ("ASURASCANS.COM",
# "www.foo.net"). Real dialogue never contains a web domain, so dropping
# domain-like blocks is safe. OCR_DROP_TEXT adds case-insensitive substrings to
# drop as well — for recurring watermarks or scene signage the CV heuristics can't
# tell from dialogue (e.g. OCR_DROP_TEXT="asurascans,room hof&soju").
_URL_RE = re.compile(r"(https?://|www\.|\b[\w-]+\.(COM|NET|ORG|IO|XYZ|INFO)\b)", re.I)
_DROP_TEXT = [s.strip().lower() for s in os.getenv("OCR_DROP_TEXT", "").split(",") if s.strip()]


def _is_watermark(text: str) -> bool:
    if _URL_RE.search(text):
        return True
    low = text.lower()
    return any(d in low for d in _DROP_TEXT)


def _title_band_bottom(lines: List["_Line"]) -> int:
    if not lines:
        return 0
    ordered = sorted(lines, key=lambda l: l.box[1])
    band = 0

    # Signal A: large stylized logo text near the top.
    logo = [l for l in ordered
            if (l.box[1] + l.box[3] / 2) < TITLE_SCAN_PX and l.box[3] > LOGO_MIN_H]
    if logo:
        band = max(l.box[1] + l.box[3] for l in logo)

    # Signal B: text confined to the top title-card zone, then a large empty gap
    # before the story begins. Only lines inside the zone count as the top group,
    # so normal inter-panel gaps further down never trigger it.
    top_zone = [l for l in ordered if l.box[1] < TITLE_SCAN_PX]
    if top_zone:
        group_bottom = max(l.box[1] + l.box[3] for l in top_zone)
        below = [l.box[1] for l in ordered if l.box[1] >= group_bottom]
        if below and min(below) - group_bottom > TITLE_GAP:
            band = max(band, group_bottom)

    return band + TITLE_MARGIN if band else 0


def _footer_band_top(lines: List["_Line"], img_h: int) -> int:
    """Y above which the end-of-episode footer begins; img_h (nothing dropped) if
    none. The footer is anchored on the series logo: large stylized lettering in
    the bottom band. Everything from the logo's top down (credits, watermark) is
    the footer.

    We deliberately use ONLY the oversized-logo signal, not a gap heuristic: a
    normal dialogue bubble can also sit alone at the bottom after a big empty gap
    (e.g. 4-223ec.webp's "GANGING UP ON A CHILD..."), and a gap rule would wrongly
    drop it. A real footer always carries the big logo; ordinary dialogue never
    does. Missing a logo-less footer is cheaper than clipping real dialogue."""
    if not lines or img_h < FOOTER_MIN_IMG_H:
        return img_h
    logo = [l for l in lines
            if (l.box[1] + l.box[3] / 2) > img_h - FOOTER_SCAN_PX and l.box[3] > LOGO_MIN_H]
    if not logo:
        return img_h
    return min(l.box[1] for l in logo) - TITLE_MARGIN


@dataclass
class _Line:
    text: str
    box: geo.Box
    confidence: float


@dataclass
class _Group:
    lines: List[_Line] = field(default_factory=list)
    bubble: Optional[Bubble] = None


def _slice_offsets(height: int, overlap: int = OVERLAP) -> List[int]:
    if height <= TILE_H:
        return [0]
    step = TILE_H - overlap
    offsets = list(range(0, height - overlap, step))
    if offsets[-1] + TILE_H < height:
        offsets.append(height - TILE_H)
    return offsets


def _ocr_all_tiles(bgr: np.ndarray) -> List[_Line]:
    """OCR each tile, but keep a detection only from the tile that OWNS it: the
    one whose non-overlap core contains the detection's vertical centre. A bubble
    straddling a boundary is thus taken from the single tile that read it whole,
    not stitched from two clipped reads."""
    engine = get_engine()
    h = bgr.shape[0]
    offsets = _slice_offsets(h, OVERLAP)
    margin = OVERLAP // 2
    raw: List[_Line] = []
    for idx, y0 in enumerate(offsets):
        tile = bgr[y0 : y0 + TILE_H]
        top_lim = 0 if idx == 0 else margin
        bot_lim = TILE_H if idx == len(offsets) - 1 else TILE_H - margin
        for tl in engine.detect(tile):
            cy = tl.box[1] + tl.box[3] / 2
            if not (top_lim <= cy < bot_lim):
                continue
            raw.append(_Line(text=tl.text, box=geo.shift(tl.box, 0, y0),
                             confidence=tl.confidence))
    return _dedup_lines(raw)


def _dedup_lines(lines: List[_Line]) -> List[_Line]:
    """Drop duplicates produced in tile-overlap zones, keep higher confidence."""
    kept: List[_Line] = []
    for ln in sorted(lines, key=lambda l: l.confidence, reverse=True):
        if not any(_is_duplicate(ln, k) for k in kept):
            kept.append(ln)
    return kept


def _is_duplicate(a: _Line, b: _Line) -> bool:
    if geo.iou(a.box, b.box) >= DEDUP_IOU:
        return True
    # A bubble on a tile boundary is read in both tiles; each read may be clipped
    # so the boxes don't align for IoU. Identical text at nearly the same height
    # (with horizontal overlap) is the same line seen twice.
    if a.text == b.text:
        ca, cb = a.box[1] + a.box[3] / 2, b.box[1] + b.box[3] / 2
        h_over = geo.area(geo.intersection(
            (a.box[0], 0, a.box[2], 1), (b.box[0], 0, b.box[2], 1)))
        if abs(ca - cb) < 1.2 * max(a.box[3], b.box[3]) and h_over > 0:
            return True
    return False


def _expanded(box: geo.Box) -> geo.Box:
    """Grow a line box by a fraction of its own height so that neighbours on the
    same text line (wide horizontal reach) and adjacent rows (small vertical
    reach) touch, while separate bubbles far apart stay disjoint."""
    x, y, w, h = box
    hx, hy = int(1.0 * h), int(0.5 * h)
    return (x - hx, y - hy, w + 2 * hx, h + 2 * hy)


def _cluster_lines(lines: List[_Line]) -> List[List[_Line]]:
    """Group text lines into reading blocks by 2-D proximity (union-find on
    expanded boxes). Handles OCR that fragments one visual line into several
    boxes (e.g. an indented leading "A"); a large gap still starts a new block,
    keeping two stacked bubbles separate even if detection merged them."""
    n = len(lines)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = [_expanded(l.box) for l in lines]
    for i in range(n):
        for j in range(i + 1, n):
            if geo.area(geo.intersection(boxes[i], boxes[j])) > 0:
                parent[find(i)] = find(j)

    clusters: dict = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(lines[i])
    return list(clusters.values())


def _reading_order(lines: List[_Line]) -> List[_Line]:
    """Order lines top-to-bottom, left-to-right, grouping fragments that share a
    row so an indented leading word isn't sorted after the rest of its line."""
    rows: List[List[_Line]] = []
    for ln in sorted(lines, key=lambda l: l.box[1]):
        cy = ln.box[1] + ln.box[3] / 2
        row = next((r for r in rows
                    if abs(cy - (r[0].box[1] + r[0].box[3] / 2)) < 0.6 * ln.box[3]), None)
        if row is None:
            row = []
            rows.append(row)
        row.append(ln)
    ordered: List[_Line] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda l: l.box[0]))
    return ordered


def _build_block(idx: int, group: _Group, img_w: int, img_h: int) -> Block:
    members = _reading_order(group.lines)
    text = " ".join(m.text for m in members)
    text = _normalize_case(_spellfix(_fix_digit_confusions(_fix_punctuation(text))))
    text_box = geo.union_box([m.box for m in members])
    conf = float(np.mean([m.confidence for m in members]))

    if group.bubble is not None:
        paint_box = group.bubble.box
        kind = BlockType(group.bubble.kind)
    else:
        paint_box = geo.pad(text_box, px=max(6, int(0.12 * text_box[3])), w_max=img_w, h_max=img_h)
        # wide caption spanning most of the strip => narration, else dialogue
        kind = BlockType.narration if text_box[2] > 0.6 * img_w else BlockType.dialogue

    return Block(
        id=idx,
        type=kind,
        text=text,
        bbox=BBox(x=paint_box[0], y=paint_box[1], w=paint_box[2], h=paint_box[3]),
        text_bbox=BBox(x=text_box[0], y=text_box[1], w=text_box[2], h=text_box[3]),
        confidence=round(conf, 4),
    )


def extract(bgr: np.ndarray) -> ExtractResponse:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    lines = _ocr_all_tiles(bgr)
    bubbles = detect_bubbles(bgr)

    # The logo (large text) is used to locate the title card / footer before we
    # discard oversized lines, so compute the bands first.
    title_bottom = _title_band_bottom(lines)
    footer_top = _footer_band_top(lines, h)

    # Keep a line if it's inside a bubble, or on a clean light background and
    # confident enough. Skip the title card / end-of-episode footer, oversized
    # SFX/logo lettering, and (later, per block) text without real words.
    kept: List[_Line] = []
    for ln in lines:
        cy = ln.box[1] + ln.box[3] / 2
        if cy < title_bottom or cy >= footer_top:
            continue
        if ln.box[3] > MAX_LINE_H:
            continue
        if find_container(ln.box, bubbles) is not None:
            kept.append(ln)
            continue
        clean, _ = background_is_clean(gray, ln.box)
        if clean and ln.confidence >= MIN_CONF_LOOSE:
            kept.append(ln)

    # Cluster first (splits stacked bubbles), then attach a bubble to each
    # cluster for its paint box / type -- but only if the bubble fits snugly.
    # A bubble far taller than its text is a merge artifact; fall back to the
    # padded text box so each cluster keeps its own region.
    groups: List[_Group] = []
    for cl in _cluster_lines(kept):
        cluster_box = geo.union_box([l.box for l in cl])
        bubble = find_container(cluster_box, bubbles)
        if bubble is not None and bubble.box[3] > BUBBLE_FIT_RATIO * cluster_box[3]:
            bubble = None
        groups.append(_Group(lines=cl, bubble=bubble))

    groups.sort(key=lambda g: min(l.box[1] for l in g.lines))
    blocks = [_build_block(i, g, w, h) for i, g in enumerate(groups)]
    # Drop blocks with no real words (episode numbers, symbol-only SFX misreads)
    # and scanlation watermarks / configured signage (URLs, OCR_DROP_TEXT).
    blocks = [b for b in blocks
              if _letter_count(b.text) >= MIN_LETTERS and not _is_watermark(b.text)]
    for new_id, b in enumerate(blocks):
        b.id = new_id

    return ExtractResponse(
        image_size=Size(width=w, height=h),
        backend=get_engine().name,
        blocks=blocks,
    )
