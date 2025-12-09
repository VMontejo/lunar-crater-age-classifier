from fastapi import FastAPI
from pydantic import BaseModel
import os

# --- 1. Define Application Metadata and Startup ---
app = FastAPI(
    title="Lunar Crater Age Classifier API",
    version="0.1.0",
    description="API for classifying lunar image chipouts into 'New Crater', 'Old Crater', or 'No Crater'."
)

# --- 2. Define Request/Response Schemas (Pydantic) ---
# A simple health response schema for demonstration
class HealthResponse(BaseModel):
    status: str
    message: str
    environment: str = os.environ.get("ENV", "local")

# --- 3. Define the Minimal Endpoint ---
@app.get("/health", response_model=HealthResponse)
def get_health():
    """
    Returns the API status.
    This is the minimal endpoint required to verify the server is running.
    """
    return {
        "status": "ok",
        "message": "Service operational.",
    }

# Note: The real '/predict' endpoint would go here, requiring image data validation.
# We will skip that complex logic for the minimal endpoint.
