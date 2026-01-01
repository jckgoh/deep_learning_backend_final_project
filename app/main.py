# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.response import PredictionResponse
from app.services.predictor import predict_image

app = FastAPI(
    title="Breast Cancer Detection API",
    description="Classifies breast tumor images as benign or malignant.",
    version="1.0.0"
)

# Allow frontend access (React, HTML, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://deep-learning-frontend-final-projec.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()

    # Run prediction
    result = predict_image(image_bytes)

    return {
        "prediction": result["class"],
        "confidence": result["confidence"]
    }

