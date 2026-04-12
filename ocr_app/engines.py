import threading

from rapidocr import RapidOCR
from ultralytics import YOLO

from .settings import CONFIG_PATH, MODEL_PATH


_thread_ctx = threading.local()


def get_worker_engines() -> tuple[YOLO, RapidOCR]:
    model = getattr(_thread_ctx, "model", None)
    if model is None:
        _thread_ctx.model = YOLO(MODEL_PATH, task="detect")
        model = _thread_ctx.model

    engine = getattr(_thread_ctx, "engine", None)
    if engine is None:
        _thread_ctx.engine = RapidOCR(config_path=CONFIG_PATH)
        engine = _thread_ctx.engine

    return model, engine
