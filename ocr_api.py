from ocr_app.api import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ocr_api:app", host="0.0.0.0", port=8000, reload=False)
