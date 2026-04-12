# YOLO OCR API

基于 **YOLO + RapidOCR + FastAPI** 的票据/文档 OCR 服务，支持图片与 PDF 批量识别，提供网页上传界面和 HTTP API。

## 功能

- 图片 OCR：上传 jpg/png/webp 等图片进行检测与识别
- PDF OCR：自动分页渲染后逐页识别
- 批量并发：单次可上传多个文件并发处理
- 两种返回模式：
  - `external`：仅返回聚合后的结构化结果（默认）
  - `detail`：返回检测框、置信度等完整明细
- 内置 UI：浏览器访问 `/` 或 `/ui` 直接上传测试

## 项目结构

```text
.
├── ocr_api.py                 # 服务启动入口
├── ocr_app/
│   ├── api.py                 # FastAPI 路由
│   ├── pipeline.py            # 检测 + OCR 主流程
│   ├── engines.py             # YOLO / RapidOCR 引擎管理
│   ├── settings.py            # 运行参数与路径
│   └── utils.py               # 图像/PDF/结果处理工具
├── ui/index.html              # 前端测试页
├── weights/base/best.pt       # YOLO 模型权重（必须）
├── default_rapidocr.yaml      # RapidOCR 配置
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 环境要求

- Python `3.10`
- 模型权重文件存在：`weights/base/best.pt`
- 建议系统：Linux / macOS（Windows 需自行处理 OpenCV 运行库）

## 本地启动

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 启动服务
python ocr_api.py
# 或
uvicorn ocr_api:app --host 0.0.0.0 --port 8000
```

启动后访问：

- UI: `http://127.0.0.1:8000/`
- 健康检查: `http://127.0.0.1:8000/health`

## Docker 启动

```bash
docker compose up -d --build
```

查看状态：

```bash
docker compose ps
docker compose logs -f ocr-api
```

说明：当前 `docker-compose.yml` 使用了项目内 `weights/` 与 `default_rapidocr.yaml` 挂载，请确保这两个路径存在。

## API

### `POST /ocr`

`multipart/form-data` 参数：

- `files`：一个或多个文件（图片或 PDF）
- `response_mode`：`external` 或 `detail`（默认 `external`）

示例（单图）：

```bash
curl -X POST "http://127.0.0.1:8000/ocr" \
  -F "files=@./demo.jpg" \
  -F "response_mode=external"
```

示例（多文件）：

```bash
curl -X POST "http://127.0.0.1:8000/ocr" \
  -F "files=@./demo1.jpg" \
  -F "files=@./demo2.pdf" \
  -F "response_mode=detail"
```

### 返回示例（`external`）

```json
{
  "source_image": "<base64_png>",
  "overall_result": {
    "invoice_code": "xxxx",
    "invoice_no": "xxxx"
  }
}
```

### 返回示例（`detail`）

```json
{
  "source_name": "demo.jpg",
  "source_image": "<base64_png>",
  "model_path": "/yolo-ocr/weights/base/best.pt",
  "overall_result": {
    "invoice_code": "xxxx"
  },
  "total": 2,
  "results": [
    {
      "class_name": "invoice_code",
      "value": "xxxx",
      "confidence": 0.93,
      "bbox": [100, 120, 280, 160]
    }
  ]
}
```

## 环境变量

- `OCR_MAX_WORKERS`：并发线程数，默认 `4`
- `PDF_DPI`：PDF 渲染 DPI，默认 `200`
- `PDF_MAX_PAGES`：单个 PDF 最大页数，默认 `200`

示例：

```bash
OCR_MAX_WORKERS=8 PDF_DPI=250 python ocr_api.py
```

## 常见问题

- 报错 `文件无法解码为图片`：请确认上传文件为有效图片/PDF。
- 报错找不到权重：确认 `weights/base/best.pt` 存在且容器内路径可读。
- PDF 处理报错：确认已安装 `pymupdf`（`requirements.txt` 已包含）。

## 推送到 GitHub

```bash
git init
git add .
git commit -m "init: yolo ocr api"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

如果你已初始化过仓库，只需要：

```bash
git add README.md
git commit -m "docs: add project README"
git push
```
