import base64
import re
from typing import Any

import cv2
import numpy as np

from .settings import PDF_DPI, PDF_MAX_PAGES


_RULE_ALIASES = {
    "after_colon": "after_colon",
    "colon_right": "after_colon",
    "keep_after_colon": "after_colon",
    "alnum": "alnum_upper",
    "alnum_upper": "alnum_upper",
    "num_alpha": "alnum_upper",
    "amount": "amount",
    "money": "amount",
    "decimal": "amount",
    "date": "date",
    "ymd": "date",
    "skip_ocr": "skip_ocr",
    "no_ocr": "skip_ocr",
    "drop_overall": "drop_overall",
    "drop_external": "drop_overall",
    "ignore_external": "drop_overall",
    "no_process": "no_process",
    "raw": "no_process",
}

_DEFAULT_CLASS_RULES: dict[str, tuple[str, ...]] = {
    "buyer_code": ("alnum_upper",),
    "buyer_name": ("after_colon",),
    "check_code": ("alnum_upper",),
    "electronic_ticket_number": ("alnum_upper",),
    "invoice_code": ("alnum_upper",),
    "invoice_number": ("alnum_upper",),
    "issue_date": ("date",),
    "machine_number": ("alnum_upper",),
    "seller_code": ("alnum_upper",),
    "seller_name": ("after_colon",),
    "tax_exclusive_total_amount": ("amount",),
    "tax_inclusive_total_amount": ("amount",),
    "tax_total_amount": ("amount",),
    "title": ("no_process",),
}

_FULLWIDTH_TRANSLATION = str.maketrans(
    "０１２３４５６７８９．，：／－（）　",
    "0123456789.,:/-() ",
)


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


def _normalize_rule_token(rule: str) -> str:
    token = rule.strip().lower().replace("-", "_").replace(" ", "_")
    return _RULE_ALIASES.get(token, token)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def parse_class_spec(class_name: str) -> tuple[str, list[str]]:
    raw = str(class_name or "").strip()
    if not raw:
        return "", []

    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
        base = parts[0]
        tokens = [token for segment in parts[1:] for token in re.split(r"[,+]", segment)]
    else:
        at_idx = raw.find("@")
        hash_idx = raw.find("#")
        idx_candidates = [idx for idx in (at_idx, hash_idx) if idx >= 0]
        if idx_candidates:
            idx = min(idx_candidates)
            base = raw[:idx].strip()
            tokens = re.split(r"[|,@#+]", raw[idx + 1 :])
        else:
            base = raw
            tokens = []

    inline_rules = _dedupe_keep_order([_normalize_rule_token(token) for token in tokens if token.strip()])
    return (base or raw), inline_rules


def _infer_default_rules(class_name: str) -> list[str]:
    explicit = list(_DEFAULT_CLASS_RULES.get(class_name, ()))
    if explicit:
        return explicit

    inferred: list[str] = []
    if class_name.endswith("_name"):
        inferred.append("after_colon")
    if class_name.endswith("_amount"):
        inferred.append("amount")
    if class_name.endswith("_date"):
        inferred.append("date")
    if class_name.endswith("_code") or class_name.endswith("_number"):
        inferred.append("alnum_upper")
    return _dedupe_keep_order(inferred)


def resolve_class_rules(class_name: str, inline_rules: list[str] | None = None) -> list[str]:
    rules = _infer_default_rules(class_name)
    if inline_rules:
        rules.extend(inline_rules)
    return _dedupe_keep_order(rules)


def should_skip_ocr(class_name: str, rules: list[str]) -> bool:
    return "skip_ocr" in rules


def should_drop_overall(class_name: str, rules: list[str]) -> bool:
    return "drop_overall" in rules


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _to_half_width(text: str) -> str:
    return text.translate(_FULLWIDTH_TRANSLATION)


def _value_after_colon(text: str) -> str:
    parts = re.split(r"[：:]", text, maxsplit=1)
    if len(parts) == 2:
        return _normalize_spaces(parts[1])
    return text


def _value_alnum_upper(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def _value_amount(text: str) -> str:
    normalized = _to_half_width(text).replace(" ", "")
    candidates = re.findall(r"\d[\d,.]*", normalized)
    if not candidates:
        fallback = re.sub(r"[^0-9.]", "", normalized)
        if fallback.count(".") > 1:
            first, *rest = fallback.split(".")
            fallback = first + "." + "".join(rest)
        return fallback.strip(".")

    best = max(candidates, key=lambda s: (sum(ch.isdigit() for ch in s), len(s)))
    best = best.replace(",", "")
    if best.count(".") > 1:
        first, *rest = best.split(".")
        best = first + "." + "".join(rest)
    return best.strip(".")


def _value_date(text: str) -> str:
    normalized = _to_half_width(text)
    match = re.search(r"(19\d{2}|20\d{2})\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})", normalized)
    if not match:
        return _normalize_spaces(normalized)

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return _normalize_spaces(normalized)
    return f"{year:04d}-{month:02d}-{day:02d}"


def normalize_value_by_rules(value: str, rules: list[str]) -> str:
    text = _normalize_spaces(_to_half_width(str(value or "")))
    if not text:
        return text

    if "no_process" in rules:
        return text

    if "after_colon" in rules:
        text = _value_after_colon(text)
    if "alnum_upper" in rules:
        text = _value_alnum_upper(text)
    if "amount" in rules:
        text = _value_amount(text)
    if "date" in rules:
        text = _value_date(text)

    return _normalize_spaces(text)


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
        if bool(item.get("exclude_from_overall", False)):
            continue

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
