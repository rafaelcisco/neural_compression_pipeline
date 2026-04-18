from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from compression.huffman.encoder import compress
from compression.huffman.decoder import decompress

# Creates the FastAPI application instance.
app = FastAPI()


# Defines the expected request body for compression.
class CompressRequest(BaseModel):
    text: str


# Defines the expected request body for decompression.
class DecompressRequest(BaseModel):
    encoded_data: str


# Returns a simple status response to confirm the service is running.
@app.get("/health")
async def health():
    return {"status": "ok"}


# Compresses input text and returns the encoded result with basic metrics.
@app.post("/compress")
async def compress_endpoint(request: CompressRequest):
    try:
        result = compress(request.text)

        return {
            "encoded_data": result["compressed_data"],
            "original_text": result["original_text"],
            "original_bits": result["original_bits"],
            "compressed_bits": result["compressed_bits"],
            "metrics": {
                "ratio": result["compression_ratio"]
            }
        }

    # Returns a server error if compression fails unexpectedly.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Decompresses encoded input data and returns the decoded text.
@app.post("/decompress")
async def decompress_endpoint(request: DecompressRequest):
    try:
        decoded_text = decompress(request.encoded_data)

        return {
            "decoded_text": decoded_text
        }

    # Returns a bad request error if the encoded input is invalid.
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Returns a server error for any other unexpected failure.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))