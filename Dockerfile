FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /yolo-ocr

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p /yolo-ocr/detect /yolo-ocr/detect/crops

EXPOSE 8000

CMD ["uvicorn", "ocr_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
