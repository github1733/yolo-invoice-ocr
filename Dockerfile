FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /yolo-ocr

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install rapidocr onnxruntime \
    && pip uninstall opencv-python -y \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p /yolo-ocr/detect /yolo-ocr/detect/crops

EXPOSE 8000

CMD ["uvicorn", "ocr_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]