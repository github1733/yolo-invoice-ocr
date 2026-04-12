import base64
from typing import Any

import cv2
import numpy as np

from .settings import PDF_DPI, PDF_MAX_PAGES


def image_to_base64_png(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("图片编码失败，无法转换为 base64")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def draw_detection(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
    cv2.rectangle(image, (x1, y1), (x2, y2), (40, 180, 99), 2)
    text_scale = 0.55
    text_thickness = 1
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
    rect_y1 = max(0, y1 - th - 8)
    rect_y2 = max(0, y1)
    rect_x2 = min(image.shape[1] - 1, x1 + tw + 8)
    cv2.rectangle(image, (x1, rect_y1), (rect_x2, rect_y2), (40, 180, 99), -1)
    cv2.putText(
        image,
        label,
        (x1 + 4, max(12, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def get_class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def collect_texts(obj: Any) -> list[str]:
    texts: list[str] = []

    if obj is None:
        return texts

    if isinstance(obj, str):
        val = obj.strip()
        if val:
            texts.append(val)
        return texts

    if hasattr(obj, "txts") and isinstance(getattr(obj, "txts"), (list, tuple)):
        for t in getattr(obj, "txts"):
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
        if texts:
            return texts

    if isinstance(obj, dict):
        if "txts" in obj and isinstance(obj["txts"], (list, tuple)):
            for t in obj["txts"]:
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
            if texts:
                return texts

        for v in obj.values():
            texts.extend(collect_texts(v))
        return texts

    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                second = item[1]
                if isinstance(second, (list, tuple)) and second:
                    first = second[0]
                    if isinstance(first, str) and first.strip():
                        texts.append(first.strip())
                        continue
                if isinstance(second, str) and second.strip():
                    texts.append(second.strip())
                    continue
            texts.extend(collect_texts(item))
        return texts

    return texts


def extract_ocr_value(ocr_result: Any) -> str:
    texts = collect_texts(ocr_result)
    merged: list[str] = []
    for txt in texts:
        if txt not in merged:
            merged.append(txt)
    return " ".join(merged)


def build_external_json(merged_json: dict[str, Any]) -> dict[str, str]:
    best_by_class: dict[str, dict[str, Any]] = {}

    for item in merged_json.get("results", []):
        class_name = str(item.get("class_name", "")).strip()
        value = str(item.get("value", "")).strip()
        confidence = float(item.get("confidence", 0.0))
        if not class_name:
            continue

        prev = best_by_class.get(class_name)
        if prev is None:
            best_by_class[class_name] = {"value": value, "confidence": confidence}
            continue

        prev_value = str(prev.get("value", "")).strip()
        prev_conf = float(prev.get("confidence", 0.0))

        if value and not prev_value:
            best_by_class[class_name] = {"value": value, "confidence": confidence}
        elif value and prev_value and confidence > prev_conf:
            best_by_class[class_name] = {"value": value, "confidence": confidence}
        elif not value and not prev_value and confidence > prev_conf:
            best_by_class[class_name] = {"value": value, "confidence": confidence}

    return {k: str(v.get("value", "")) for k, v in best_by_class.items()}


def is_pdf_file(file_name: str, content: bytes) -> bool:
    lower_name = file_name.lower()
    if lower_name.endswith(".pdf"):
        return True
    return content[:5] == b"%PDF-"


def pdf_bytes_to_images(content: bytes, dpi: int = PDF_DPI) -> list[np.ndarray]:
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "PDF 识别依赖 PyMuPDF，请先安装: pip install pymupdf"
        ) from e

    images: list[np.ndarray] = []
    scale = max(0.1, float(dpi) / 72.0)
    matrix = fitz.Matrix(scale, scale)

    doc = fitz.open(stream=content, filetype="pdf")
    try:
        page_count = doc.page_count
        if page_count <= 0:
            raise ValueError("PDF 没有可识别页面")
        if page_count > PDF_MAX_PAGES:
            raise ValueError(f"PDF 页数超过限制: {page_count} > {PDF_MAX_PAGES}")

        for i in range(page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            images.append(image)
    finally:
        doc.close()

    return images
