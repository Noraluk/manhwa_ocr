import cv2
import numpy as np

from app.detector import background_is_clean, detect_bubbles


def _blank(h=600, w=600, val=245):
    return np.full((h, w, 3), val, np.uint8)


def test_detects_white_bubble_with_text_on_dark_art():
    img = _blank(val=40)  # dark artwork background
    cv2.ellipse(img, (300, 300), (180, 110), 0, 0, 360, (255, 255, 255), -1)
    cv2.putText(img, "HELLO", (200, 315), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
    bubbles = detect_bubbles(img)
    assert len(bubbles) == 1
    assert bubbles[0].kind == "dialogue"


def test_rectangular_caption_box_classified_as_narration():
    img = _blank(val=40)
    cv2.rectangle(img, (120, 250), (480, 350), (255, 255, 255), -1)
    cv2.putText(img, "CAPTION", (150, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    bubbles = detect_bubbles(img)
    assert bubbles and bubbles[0].kind == "narration"


def test_pure_white_blob_without_text_is_ignored():
    img = _blank(val=40)
    cv2.circle(img, (300, 300), 120, (255, 255, 255), -1)  # no text inside
    assert detect_bubbles(img) == []


def test_background_clean_on_paper_but_not_on_dark_art():
    paper = _blank(val=250)
    cv2.putText(paper, "hi", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    clean, mean = background_is_clean(cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY), (30, 20, 120, 60))
    assert clean and mean > 200

    noisy = np.random.randint(0, 90, (100, 200, 3), np.uint8)
    clean2, _ = background_is_clean(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY), (20, 20, 120, 40))
    assert not clean2
