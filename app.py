import os
import tempfile
import base64
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

try:
    from ocr.model.predict import predict as ocr_predict
    from compression.huffman.encoder import compress as huffman_compress
    from compression.huffman.decoder import decompress as huffman_decompress
except ImportError as e:
    print(f"Import error: {e}")

app = FastAPI(title="Neural Compression Pipeline API")

class PipelineResponse(BaseModel):
    ocr_result: dict
    compression_result: dict
    decompressed_text: str

@app.get("/")
async def root():
    return {
        "message": "Neural Compression Pipeline is online",
        "endpoints": {
            "health": "/health",
            "pipeline": "/pipeline (POST)",
            "documentation": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Pipeline orchestrator is running"}

@app.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(file: UploadFile = File(...)):
    """
    Full Pipeline:
    1. Receive Image
    2. OCR Prediction
    3. Huffman Compression
    4. Huffman Decompression
    """
    
    # 1. Read and save image to temp file
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded file: {str(e)}")

    try:
        # 2. OCR Prediction
        weights_path = os.path.join("ocr", "model", "best_model.pth")
        if not os.path.exists(weights_path):
            raise HTTPException(status_code=500, detail=f"Model weights not found at {weights_path}")
        
        prediction_data = ocr_predict(tmp_path, weights_path)
        
        extracted_text = str(prediction_data["prediction"])
        
        # 3. Huffman Compression (Encode)
        comp_res = huffman_compress(extracted_text)
        
        # 4. Huffman Decompression (Decode)
        decoded_text = huffman_decompress(comp_res["compressed_data"])
        
        return {
            "ocr_result": prediction_data,
            "compression_result": {
                "encoded_bits": comp_res["compressed_data"],
                "original_bits": comp_res["original_bits"],
                "compressed_bits": comp_res["compressed_bits"],
                "ratio": comp_res["compression_ratio"]
            },
            "decompressed_text": decoded_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    
    finally:
        # Clear temp files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
