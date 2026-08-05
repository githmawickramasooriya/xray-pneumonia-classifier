import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model_utils import predict_and_explain, validate_image

app = FastAPI(title="Pneumonia X-Ray Risk Awareness Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(status_code=400, content={"error": "Please upload a JPG or PNG image."})

    file_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{file_id}_input.jpg")
    output_path = os.path.join(TEMP_DIR, f"{file_id}_overlay.png")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    is_valid, reason = validate_image(input_path)
    if not is_valid:
        os.remove(input_path)
        return JSONResponse(status_code=400, content={"error": reason})

    try:
        result = predict_and_explain(input_path, output_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {str(e)}"})

    result["overlay_url"] = f"/overlay/{file_id}"
    return result


@app.get("/overlay/{file_id}")
def get_overlay(file_id: str):
    path = os.path.join(TEMP_DIR, f"{file_id}_overlay.png")
    return FileResponse(path, media_type="image/png")