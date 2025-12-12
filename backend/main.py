from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image

from backend.model_loader import predict_image, explain_image

app = FastAPI()

import threading
import time
import requests

RENDER_URL = "https://alzheimers-prediction-deployment.netlify.app/"

def keep_alive():
    while True:
        try:
            requests.get(RENDER_URL, timeout=10)
        except Exception:
            pass
        time.sleep(300)  # 5 minutes = 300 seconds

# Start background keep-alive thread
threading.Thread(target=keep_alive, daemon=True).start()
# 🔥 FIXED — allow Netlify + Render frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*", 
        "https://alzheimers-prediction-deployment.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Backend running for Alzheimer's prediction"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    result = predict_image(image, file.filename, contents)

    return result

@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    heatmap_bytes = explain_image(image)

    return StreamingResponse(
        BytesIO(heatmap_bytes),
        media_type="image/png"
    )




