from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image

from model_loader import predict_image, explain_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Backend running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    return predict_image(image)

@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    heatmap = explain_image(image)
    return StreamingResponse(BytesIO(heatmap), media_type="image/png")
