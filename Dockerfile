# Manhwa OCR API — container image tuned for Hugging Face Spaces (Docker SDK).
# HF runs the container as a non-root user (uid 1000) and routes to app_port
# (7860 here, set in README.md frontmatter). CPU-only; PaddleOCR default backend.
FROM python:3.10-slim

# System libs: libglib2.0-0 + libgl1 for OpenCV (paddleocr pulls in the non-headless
# opencv-contrib-python, which needs libGL.so.1), libgomp1 for Paddle's OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user so writes (model cache, tmp) land in a writable HOME on HF.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
WORKDIR /home/user/app

COPY --chown=user requirements-slim.txt .
RUN pip install --no-cache-dir --user -r requirements-slim.txt

COPY --chown=user . .

# Bake the Paddle det+rec models into the image (into $HOME/.paddlex) so the first
# request after a cold start doesn't wait on a download. A tiny predict forces the
# fetch; config must match app/ocr.py::_PaddleOCR (orientation off).
RUN python -c "import numpy as np; from paddleocr import PaddleOCR; \
o=PaddleOCR(lang='en', use_textline_orientation=False, use_doc_orientation_classify=False, use_doc_unwarping=False); \
o.predict(input=np.full((64,128,3),255,'uint8'))"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
