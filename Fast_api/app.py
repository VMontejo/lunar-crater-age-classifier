from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from lunar_crater_age_logic.preprocess import preprocess_image
from lunar_crater_age_logic.grad_cam import make_gradcam_heatmap
import base64

# --- 1. Constants and Global Variables ---
# NOTE: Adjust this path when not local
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),"models/santanu_best_model.keras")
TARGET_SIZE = (227, 227)
CLASS_NAMES = ["New Crater (0)", "Old Crater (1)", "No Crater (2)"]
LAST_CONV_LAYER = "conv2d_9"
global model

# Helper function to encode image to base64
def encode_image_to_base64(image_data: np.ndarray) -> str:
    """
    Converts a raw heatmap (float or uint8) array into a Base64-encoded PNG string.
    Ensures the array is properly formatted (uint8, 3-channel) for PIL.
    """
    heatmap_array = image_data.copy()
    ## 1. Ensure array is normalized floats (0.0 to 1.0)
    # Handle both common cases: uint8 (0-255) or float32 (0.0-1.0)
    if heatmap_array.dtype != np.float32 and heatmap_array.dtype != np.float64:
        heatmap_array = heatmap_array.astype(np.float32) / 255.0

    # 2. Handle shape: Ensure it is a 3-channel array (H, W, 3)
    # Grad-CAM often returns (H, W, 1) or just (H, W)
    if heatmap_array.ndim == 2:
        # Convert (H, W) to (H, W, 3) by stacking the grayscale channel
        heatmap_array = np.stack([heatmap_array] * 3, axis=-1)
    elif heatmap_array.shape[-1] == 1:
        # Convert (H, W, 1) to (H, W, 3)
        heatmap_array = np.concatenate([heatmap_array] * 3, axis=-1)

    # 3. Convert to uint8 (0-255) for PIL
    # Clip to 0-1 range just in case of overshoots from calculations
    heatmap_array = np.clip(heatmap_array, 0.0, 1.0)
    img_data_uint8 = (heatmap_array * 255).astype(np.uint8)

    # 4. Convert to PIL Image
    img = Image.fromarray(img_data_uint8, mode='RGB')

    # 5. Save image to buffer and encode
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def encode_image_to_base64_rgb(image: np.ndarray) -> str:
    """
    Encode an RGB uint8 image directly to base64 PNG (NO normalization).
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    img = Image.fromarray(image, mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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
    heatmap_image: str
    overlay_image: str

# --- 3. FastAPI Application Definition ---
app = FastAPI(
    title="Lunar Crater Age Classifier API",
    version="0.0.2",
    description="API for classifying lunar image chipouts."
)

@app.get("/")
async def root():
    return {"message": "Lunar Crater Age Classifier API", "docs": "/docs"}

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

        # --- TEMPORARY DEBUGGING BLOCK TO GET CONV LAYER NAME ---
        print("\n--- Model Layers for Grad-CAM ---")
        # Print all layer names in the model
        for layer in model.layers:
            # We are typically interested in the last Conv2D or MaxPool layer before dense layers.
            if 'conv' in layer.name or 'pool' in layer.name:
                print(f"Layer Name: {layer.name}, Type: {type(layer).__name__}")
        print("---------------------------------")
        # --- END TEMPORARY DEBUGGING BLOCK TO GET CONV LAYER NAME---

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

@app.post("/predict", response_model=PredictionResponse, operation_id="predict")
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

        # 2. Convert PIL → TensorFlow tensor (uint8)
        image_np = np.array(image, dtype=np.uint8)
        image_tf = tf.convert_to_tensor(image_np)

        # 3. Apply Preprocessing function (Z-score normalization)
        preprocessed_image = preprocess_image(image_tf,
                                              model_type="custom",
                                              normalization="zscore"
                                              )

        # 3.5. Ensure the tensor is float32, which is required by model.predict and Grad-CAM
        if preprocessed_image.dtype != tf.float32:
            preprocessed_image = tf.cast(preprocessed_image, tf.float32)

        # 4. Make Prediction and index for Grad-CAM
        predictions = model.predict(preprocessed_image)
        predicted_index = np.argmax(predictions[0])

        # 5. Gradient-weighted Class Activation Mapping heatmap generation
        # Adding explainability via Grad-CAM
        overlay, heatmap_array = make_gradcam_heatmap(
            preprocessed_image,
            model,
            LAST_CONV_LAYER,
            original_image = image_np,
            pred_index=predicted_index
        ) # The output expected to be a NumPy array representing the heatmap image

        # 6. Encode Heatmap
        encoded_heatmap = encode_image_to_base64(heatmap_array)
        encoded_overlay = encode_image_to_base64_rgb(overlay)

        # 7. Process Softmax Output
        predicted_proba = np.max(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]

        # 8. Abstention Check (recommended based on LROCNet)
        # Insert confidence threshold (e.g., 0.8) here:
        # if predicted_proba < 0.8:
        #     return {"class_name": "Abstain", "confidence": predicted_proba, ...}

        return {
            "class_name": predicted_class,
            "class_index": int(predicted_index),
            "confidence": float(predicted_proba),
            "message": f"Successfully classified image: {file.filename}",
            "overlay_image": encoded_overlay,
            "heatmap_image": encoded_heatmap
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal error: {e}")
