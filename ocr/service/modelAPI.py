from pydantic import BaseModel
import base64
import os
import tempfile
from PIL import Image
import io
from ocr.model.predict import predict as ocr_predict
from ocr.model.cnn import OCRCNN

app = FastAPI()

# Define expected request body
class PredictRequest(BaseModel):
    image_base64: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(request: PredictRequest):
    try:
        # Decode the base64 image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # Create temp file to save the imagecd 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image_path = tmp_file.name
            image.save(image_path)

        # Define the path to the model weights
        model_weights_path = "ocr/model/best_model.pth"

        # Predict
        prediction_result = ocr_predict(image_path, model_weights_path)

        # Delete temp file
        os.remove(image_path)

        return prediction_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))