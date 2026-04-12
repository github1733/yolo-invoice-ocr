import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
UI_INDEX = BASE_DIR / "ui" / "index.html"

MODEL_PATH = str(BASE_DIR / "weights" / "base" / "best.pt")
CONFIG_PATH = "default_rapidocr.yaml"

BOX_PADDING = 4
MIN_CROP_HEIGHT = 64
MAX_UPSCALE = 4.0

MAX_WORKERS = max(1, int(os.getenv("OCR_MAX_WORKERS", "4")))
PDF_DPI = int(os.getenv("PDF_DPI", "200"))
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "200"))
