from typing import Any

import cv2
import numpy as np
from rapidocr import RapidOCR
from ultralytics import YOLO

from .engines import get_worker_engines
from .settings import BOX_PADDING, MAX_UPSCALE, MIN_CROP_HEIGHT, MODEL_PATH
from .utils import (
    build_external_json,
    draw_detection,
    extract_ocr_value,
    get_class_name,
    image_to_base64_png,
    is_pdf_file,
    normalize_value_by_rules,
    pdf_bytes_to_images,
    parse_class_spec,
    resolve_class_rules,
    should_drop_overall,
    should_skip_ocr,
)


def run_detection_and_ocr_image(
    image: np.ndarray,
    source_name: str,
    model: YOLO,
    engine: RapidOCR,
) -> dict[str, Any]:
    vis_image = image.copy()
    merged_json: dict[str, Any] = {
        "source_name": source_name,
        "source_image": "",
        "model_path": MODEL_PATH,
        "overall_result": {},
        "total": 0,
        "results": [],
    }

    results = model(source=image, save=False, verbose=False)

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        orig_img = result.orig_img
        if orig_img is None:
            continue

        h, w = orig_img.shape[:2]
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, min(x1 - BOX_PADDING, w - 1))
            y1 = max(0, min(y1 - BOX_PADDING, h - 1))
            x2 = max(0, min(x2 + BOX_PADDING, w))
            y2 = max(0, min(y2 + BOX_PADDING, h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = orig_img[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            if ch == 0 or cw == 0:
                continue

            if ch < MIN_CROP_HEIGHT:
                scale = min(MAX_UPSCALE, MIN_CROP_HEIGHT / float(ch))
                new_w = max(1, int(round(cw * scale)))
                new_h = max(1, int(round(ch * scale)))
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            class_name_raw = get_class_name(result.names, cls_id)
            class_name, inline_rules = parse_class_spec(class_name_raw)
            rules = resolve_class_rules(class_name, inline_rules)

            label = f"{class_name} {float(conf):.2f}"
            draw_detection(vis_image, int(x1), int(y1), int(x2), int(y2), label)

            raw_value = ""
            ocr_error = None
            try:
                if not should_skip_ocr(class_name, rules):
                    ocr_raw = engine(crop)
                    raw_value = extract_ocr_value(ocr_raw)
            except Exception as e:  # noqa: BLE001
                ocr_error = str(e)

            value = normalize_value_by_rules(raw_value, rules)
            item: dict[str, Any] = {
                "class_name": class_name,
                "value": value,
                "confidence": round(float(conf), 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }
            if class_name_raw != class_name:
                item["class_name_raw"] = class_name_raw
            if raw_value and raw_value != value:
                item["value_raw"] = raw_value
            if should_drop_overall(class_name, rules):
                item["exclude_from_overall"] = True
            if ocr_error:
                item["ocr_error"] = ocr_error

            merged_json["results"].append(item)

    merged_json["total"] = len(merged_json["results"])
    merged_json["overall_result"] = build_external_json(merged_json)
    merged_json["source_image"] = image_to_base64_png(vis_image)
    return merged_json


def process_pdf_file(file_name: str, content: bytes, response_mode: str) -> dict[str, Any]:
    model, engine = get_worker_engines()
    page_images = pdf_bytes_to_images(content)

    pages: list[dict[str, Any]] = []
    for page_idx, image in enumerate(page_images, start=1):
        page_source_name = f"{file_name}#page_{page_idx}"
        detailed = run_detection_and_ocr_image(image, page_source_name, model, engine)

        if response_mode == "detail":
            pages.append(
                {
                    "page_index": page_idx,
                    "data": detailed,
                }
            )
        else:
            pages.append(
                {
                    "page_index": page_idx,
                    "data": {
                        "source_image": detailed.get("source_image", ""),
                        "overall_result": detailed.get("overall_result", {}),
                    },
                }
            )

    return {
        "source_file": file_name,
        "file_type": "pdf",
        "page_count": len(pages),
        "pages": pages,
    }


def process_one_file(file_name: str, content: bytes, response_mode: str) -> Any:
    if is_pdf_file(file_name, content):
        return process_pdf_file(file_name, content, response_mode)

    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"文件无法解码为图片: {file_name}")

    model, engine = get_worker_engines()
    detailed = run_detection_and_ocr_image(image, file_name, model, engine)

    if response_mode == "detail":
        return detailed
    return {
        "source_image": detailed.get("source_image", ""),
        "overall_result": detailed.get("overall_result", {}),
    }
