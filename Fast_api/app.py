from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

# --- 1. Constants and Global Variables ---
# NOTE: Adjust this path when not local
MODEL_PATH = "notebooks/grace/grace_best_model.keras"
TARGET_SIZE = (227, 227)
CLASS_NAMES = ["New Crater (0)", "Old Crater (1)", "No Crater (2)"]
global model

# --- 2. Define Request/Response Schemas (Pydantic) ---
class HealthResponse(BaseModel):
    status: str
    message: str
    environment: str = os.environ.get("ENV", "local")

class PredictionResponse(BaseModel):
    class_name: str
    class_index: int
    confidence: float
    message: str

# --- 3. FastAPI Application Definition ---
app = FastAPI(
    title="Lunar Crater Age Classifier API",
    version="0.1.0",
    description="API for classifying lunar image chipouts."
)

# --- 4. Startup Event: Load Model ---
# This ensures the model is loaded only once when the server starts.
@app.on_event("startup")
async def startup_event():
    """Loads grace_best_model.keras artifact into memory."""
    global model
    try:
        # Use os.path.abspath to ensure the path is resolved correctly
        # relative to where 'uvicorn' process is running (project root).
        model_full_path = os.path.abspath(MODEL_PATH)
        model = tf.keras.models.load_model(model_full_path, compile=False)
        print(f"✅ Successfully loaded model from: {model_full_path}")
    except Exception as e:
        # If the model fails to load, the API should not start.
        raise RuntimeError(f"🚨 FAILED TO LOAD MODEL from {MODEL_PATH}: {e}")

# --- 5. Endpoints ---
@app.get("/health", response_model=HealthResponse)
def get_health():
    """Returns the API status and checks if the model is loaded."""
    model_status = "loaded" if 'model' in globals() else "unloaded (error)"
    return {
        "status": "ok",
        "message": f"Service operational. Model status: {model_status}",
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_crater_age(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the predicted crater age class.
    """
    if 'model' not in globals():
        raise HTTPException(status_code=500, detail="Model not initialized. Server startup failed.")

    try:
        # 1. Read the uploaded file into a PIL Image object
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 2. Convert to NumPy array and resize/expand dimensions
        # VGG16 expects (Height, Width, Channels), but the prediction model expects
        # a batch: (1, Height, Width, Channels)
        image = image.resize(TARGET_SIZE)
        image_array = np.asarray(image).astype(np.float32)
        image_expanded = np.expand_dims(image_array, axis=0)

        # 3. Apply VGG16 Preprocessing
        # Note: We have to ensure the input is 0-255 before calling this
        preprocessed_image = vgg16_preprocess_input(image_expanded)

        # 4. Make Prediction
        predictions = model.predict(preprocessed_image)

        # 5. Process Softmax Output
        predicted_proba = np.max(predictions[0])
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]

        # 6. Abstention Check (recommended based on LROCNet)
        # Insert confidence threshold (e.g., 0.8) here:
        # if predicted_proba < 0.8:
        #     return {"class_name": "Abstain", "confidence": predicted_proba, ...}

        return {
            "class_name": predicted_class,
            "class_index": int(predicted_index),
            "confidence": float(predicted_proba),
            "message": f"Successfully classified image: {file.filename}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal error: {e}")
