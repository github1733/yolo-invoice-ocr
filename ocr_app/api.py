import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from .pipeline import process_one_file
from .settings import MAX_WORKERS, UI_INDEX


_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
logger = logging.getLogger("uvicorn.error")
FAVICON_PATH = Path(UI_INDEX).parent / "favicon.ico"


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        _executor.shutdown(wait=False)


app = FastAPI(title="YOLO INVOICE OCR API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_failed method=%s path=%s cost_ms=%.2f",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "request method=%s path=%s status=%s cost_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/")
def index() -> FileResponse:
    if not UI_INDEX.exists():
        raise HTTPException(status_code=404, detail=f"UI 文件不存在: {UI_INDEX}")
    return FileResponse(UI_INDEX)


@app.get("/ui")
def ui() -> FileResponse:
    return index()


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    if FAVICON_PATH.exists():
        return FileResponse(FAVICON_PATH)
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "max_workers": MAX_WORKERS}


@app.post("/ocr")
async def ocr_api(
    files: list[UploadFile] = File(...),
    response_mode: Literal["external", "detail"] = "external",
) -> Any:
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个文件")

    prepared: list[tuple[str, bytes, int]] = []
    for i, f in enumerate(files, start=1):
        content = await f.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"空文件: {f.filename or i}")
        name = f.filename or f"file_{i}.jpg"
        prepared.append((name, content, i))

    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            _executor,
            process_one_file,
            name,
            content,
            response_mode,
        )
        for name, content, _idx in prepared
    ]

    outputs = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, out in enumerate(outputs, start=1):
        if isinstance(out, Exception):
            file_name = prepared[idx - 1][0]
            raise HTTPException(status_code=400, detail=f"处理失败({file_name}): {out}")

    if len(outputs) == 1:
        return outputs[0]
    return outputs
