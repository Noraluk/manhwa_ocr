"""API + pipeline tests with a fake OCR engine so they run fast (no torch/models).

The single endpoint is POST /ocr (NDJSON stream). Pipeline-behaviour cases post
one image and read its single result line via the `_extract` helper.
"""
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
import app.ocr as ocr_mod
import app.pipeline as pipeline
from app.ocr import TextLine


class _FakeEngine:
    name = "fake"

    def __init__(self, lines=None):
        self._lines = lines or []

    def detect(self, bgr):
        return list(self._lines)


@pytest.fixture
def client(monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setattr(pipeline, "get_engine", lambda: engine)
    monkeypatch.setattr(main, "get_engine", lambda: engine)
    ocr_mod.get_engine.cache_clear()
    return TestClient(main.app), engine


def _png_bytes(w=200, h=200, val=245):
    arr = np.full((h, w, 3), val, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _ndjson(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def _extract(c, png, filename="x.png"):
    """POST a single image to /ocr and return its OcrItem dict (status ok|error).
    On ok, item['result'] has the same shape the old /extract returned."""
    r = c.post("/ocr", files={"files": (filename, png, "image/png")})
    assert r.status_code == 200
    lines = _ndjson(r)
    assert lines[0] == {"status": "started", "total": 1}
    items = [l for l in lines if l.get("status") in ("ok", "error")]
    assert len(items) == 1
    return items[0]


def _blocks(c, png, **kw):
    item = _extract(c, png, **kw)
    assert item["status"] == "ok", item.get("error")
    return item["result"]["blocks"]


# --- endpoint behaviour -----------------------------------------------------

def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok", "backend": "fake"}


def test_no_files_rejected(client):
    c, _ = client
    # multipart with an empty "files" list isn't valid; FastAPI 422s on missing field
    assert c.post("/ocr").status_code == 422


def test_empty_file_streams_error_item(client):
    c, _ = client
    item = _extract(c, b"")
    assert item["status"] == "error" and item["error"]


def test_non_image_streams_error_item(client):
    c, _ = client
    item = _extract(c, b"not-an-image")
    assert item["status"] == "error" and item["error"]


def test_blank_image_returns_no_blocks(client):
    c, _ = client
    item = _extract(c, _png_bytes())
    assert item["status"] == "ok"
    assert item["result"]["blocks"] == []
    assert item["result"]["image_size"] == {"width": 200, "height": 200}


def test_text_on_paper_is_extracted(client):
    c, engine = client
    engine._lines = [
        TextLine("HELLO", [[20, 20], [120, 20], [120, 50], [20, 50]], 0.9),
        TextLine("WORLD", [[20, 55], [120, 55], [120, 85], [20, 85]], 0.9),
    ]
    blocks = _blocks(c, _png_bytes(val=250))
    assert len(blocks) == 1
    assert blocks[0]["text"] == "HELLO WORLD"


def test_ocr_streams_all_images_and_isolates_errors(client):
    c, engine = client
    engine._lines = [TextLine("HELLO", [[20, 20], [120, 20], [120, 50], [20, 50]], 0.9)]
    files = [
        ("files", ("p1.png", _png_bytes(val=250), "image/png")),
        ("files", ("bad.png", b"not-an-image", "image/png")),
        ("files", ("p2.png", _png_bytes(val=250), "image/png")),
    ]
    r = c.post("/ocr", files=files)
    assert r.status_code == 200
    lines = _ndjson(r)
    assert lines[0] == {"status": "started", "total": 3}
    items = [l for l in lines if l.get("status") in ("ok", "error")]
    assert [it["index"] for it in items] == [0, 1, 2]
    assert [it["status"] for it in items] == ["ok", "error", "ok"]
    assert items[0]["result"]["blocks"][0]["text"] == "HELLO"
    assert items[1]["error"]
    assert items[2]["filename"] == "p2.png"


def test_too_many_files_rejected(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(main, "MAX_FILES", 2)
    files = [("files", (f"p{i}.png", _png_bytes(), "image/png")) for i in range(3)]
    assert c.post("/ocr", files=files).status_code == 413


# --- pipeline behaviour (via a single /ocr image) ---------------------------

def test_indented_line_fragment_is_kept_and_ordered(client):
    c, engine = client
    # OCR often splits an indented leading word into its own box; it must join
    # the line (not become a 1-letter block that MIN_LETTERS drops) and sort first.
    engine._lines = [
        TextLine("GANGING UP ON", [[100, 40], [300, 40], [300, 72], [100, 72]], 0.8),
        TextLine("CHILD LIKE", [[85, 80], [340, 80], [340, 116], [85, 116]], 0.98),
        TextLine("A", [[60, 86], [78, 86], [78, 108], [60, 108]], 1.0),  # indented, left
    ]
    blocks = _blocks(c, _png_bytes(w=400, h=300, val=250))
    assert len(blocks) == 1
    assert blocks[0]["text"] == "GANGING UP ON A CHILD LIKE"
    assert blocks[0]["text_bbox"]["x"] <= 60  # box reaches the indented "A"


def test_digit_letter_confusion_is_fixed(client):
    c, engine = client
    engine._lines = [
        TextLine("5O", [[20, 20], [70, 20], [70, 52], [20, 52]], 0.6),
        TextLine("HE", [[80, 20], [130, 20], [130, 52], [80, 52]], 0.9),
    ]
    assert _blocks(c, _png_bytes(val=250))[0]["text"] == "SO HE"


def test_midword_quote_becomes_apostrophe(client):
    c, engine = client
    engine._lines = [TextLine('IT"S', [[20, 20], [90, 20], [90, 52], [20, 52]], 0.9)]
    assert _blocks(c, _png_bytes(val=250))[0]["text"] == "IT'S"


def test_semicolon_becomes_comma(client):
    c, engine = client
    engine._lines = [
        TextLine("UGH", [[20, 20], [80, 20], [80, 52], [20, 52]], 0.9),
        TextLine(";", [[82, 20], [92, 20], [92, 52], [82, 52]], 0.9),
    ]
    text = " ".join(b["text"] for b in _blocks(c, _png_bytes(val=250)))
    assert ";" not in text and "," in text


def test_trailing_colon_becomes_period_but_time_kept(client):
    c, engine = client
    engine._lines = [
        TextLine("ITSELF:", [[20, 20], [140, 20], [140, 52], [20, 52]], 0.9),
        TextLine("3:00", [[20, 70], [90, 70], [90, 102], [20, 102]], 0.9),  # time
    ]
    text = " ".join(b["text"] for b in _blocks(c, _png_bytes(val=250)))
    assert "ITSELF." in text and "3:00" in text


def test_spellcheck_fixes_nonword_but_keeps_real_words(client):
    c, engine = client
    engine._lines = [
        TextLine("GLYS", [[20, 20], [90, 20], [90, 52], [20, 52]], 0.7),   # -> GUYS
        TextLine("DEMON", [[20, 70], [110, 70], [110, 102], [20, 102]], 0.9),  # real word
    ]
    text = " ".join(b["text"] for b in _blocks(c, _png_bytes(val=250)))
    assert "GUYS" in text and "GLYS" not in text
    assert "DEMON" in text


def test_real_numbers_and_ordinals_are_preserved(client):
    c, engine = client
    engine._lines = [
        TextLine("YEAR", [[20, 20], [90, 20], [90, 52], [20, 52]], 0.9),
        TextLine("4000", [[100, 20], [180, 20], [180, 52], [100, 52]], 0.9),  # pure number
        TextLine("1ST", [[20, 70], [70, 70], [70, 102], [20, 102]], 0.9),     # ordinal
    ]
    text = " ".join(b["text"] for b in _blocks(c, _png_bytes(val=250)))
    assert "4000" in text and "1ST" in text


def test_episode_number_and_symbol_sfx_dropped(client):
    c, engine = client
    engine._lines = [
        TextLine("64", [[380, 40], [440, 40], [440, 96], [380, 96]], 0.9),   # episode no.
        TextLine("HELLO", [[20, 120], [160, 120], [160, 160], [20, 160]], 0.9),
    ]
    texts = [b["text"] for b in _blocks(c, _png_bytes(h=400, val=250))]
    assert texts == ["HELLO"]


def test_oversized_sfx_lettering_dropped(client):
    c, engine = client
    engine._lines = [
        TextLine("BOOM", [[20, 1100], [400, 1100], [400, 1340], [20, 1340]], 0.9),  # h=240
        TextLine("okay", [[20, 1400], [160, 1400], [160, 1440], [20, 1440]], 0.9),
    ]
    texts = [b["text"] for b in _blocks(c, _png_bytes(h=1600, val=250))]
    assert texts == ["OKAY"]


def test_title_card_band_dropped(client):
    c, engine = client
    engine._lines = [
        TextLine("LOGO", [[20, 30], [500, 30], [500, 230], [20, 230]], 0.9),   # big logo, top
        TextLine("by someone", [[20, 260], [300, 260], [300, 300], [20, 300]], 0.9),  # credits
        TextLine("real dialogue here", [[20, 1400], [400, 1400], [400, 1450], [20, 1450]], 0.9),
    ]
    texts = [b["text"] for b in _blocks(c, _png_bytes(h=1600, val=250))]
    assert texts == ["REAL DIALOGUE HERE"]


def test_short_image_keeps_bottom_text_no_footer(client):
    c, engine = client
    engine._lines = [
        TextLine("bottom line", [[20, 1400], [300, 1400], [300, 1450], [20, 1450]], 0.9),
    ]
    texts = [b["text"] for b in _blocks(c, _png_bytes(w=800, h=1600, val=250))]
    assert texts == ["BOTTOM LINE"]


def test_url_watermark_block_dropped(client):
    c, engine = client
    engine._lines = [
        TextLine("ASURASCANS.COM", [[600, 20], [790, 20], [790, 50], [600, 50]], 0.99),
        TextLine("HELLO THERE", [[20, 400], [220, 400], [220, 430], [20, 430]], 0.9),
    ]
    texts = [b["text"] for b in _blocks(c, _png_bytes(w=800, h=600, val=250))]
    assert texts == ["HELLO THERE"]


def test_low_confidence_text_on_art_is_dropped_as_sfx(client):
    c, engine = client
    engine._lines = [TextLine("Z", [[10, 10], [40, 10], [40, 40], [10, 40]], 0.03)]
    dark = np.random.randint(0, 80, (200, 200, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(dark).save(buf, format="PNG")
    assert _blocks(c, buf.getvalue()) == []


# --- pipeline helpers (called directly) -------------------------------------

def _mk_line(text, y, h, x=20, w=300, conf=0.9):
    return pipeline._Line(text=text, box=(x, y, w, h), confidence=conf)


def test_footer_band_detects_bottom_logo_and_credits():
    lines = [
        _mk_line("real dialogue here", 1500, 60),
        _mk_line("more dialogue", 3000, 60),
        _mk_line("SERIES LOGO", 6000, 160, w=640),
        _mk_line("credit watermark", 6300, 45, x=400, w=340, conf=0.4),
    ]
    top = pipeline._footer_band_top(lines, img_h=6800)
    assert 3060 < top <= 6000
    dropped = [l.text for l in lines if l.box[1] + l.box[3] / 2 >= top]
    kept = [l.text for l in lines if l.box[1] + l.box[3] / 2 < top]
    assert dropped == ["SERIES LOGO", "credit watermark"]
    assert kept == ["real dialogue here", "more dialogue"]


def test_footer_band_off_on_short_image():
    lines = [_mk_line("bottom line", 1400, 50)]
    assert pipeline._footer_band_top(lines, img_h=1600) == 1600


def test_footer_band_keeps_low_dialogue_without_logo():
    # 4-223ec.webp regression: normal-sized dialogue alone at the bottom after a
    # big gap (no logo) must NOT be treated as a footer.
    lines = [
        _mk_line("upper dialogue", 8000, 60),
        _mk_line("ganging up on", 13514, 30),
        _mk_line("a child like a couple", 13543, 30),
        _mk_line("of cowards", 13572, 33),
    ]
    assert pipeline._footer_band_top(lines, img_h=14222) == 14222


def test_footer_band_no_gap_keeps_bottom_text():
    lines = [_mk_line("line a", 5000, 60), _mk_line("line b", 5500, 60),
             _mk_line("line c", 6000, 60), _mk_line("line d", 6200, 60)]
    assert pipeline._footer_band_top(lines, img_h=6800) == 6800


def test_is_watermark_url_and_droptext(monkeypatch):
    assert pipeline._is_watermark("ASURASCANS.COM")
    assert pipeline._is_watermark("visit www.foo.net now")
    assert not pipeline._is_watermark("I CAN'T LEAVE THIS TO CHANCE")
    assert not pipeline._is_watermark("GO HOME.")          # not a domain
    monkeypatch.setattr(pipeline, "_DROP_TEXT", ["room hof&soju"])
    assert pipeline._is_watermark("ROOM HOF&SOJU")
