from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image

from model_loader import predict_image, explain_image

app = FastAPI()

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
    try:
        img_bytes = await file.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        result = predict_image(image)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    heatmap_bytes = explain_image(image)

    return StreamingResponse(
        BytesIO(heatmap_bytes),
        media_type="image/png"
    )
