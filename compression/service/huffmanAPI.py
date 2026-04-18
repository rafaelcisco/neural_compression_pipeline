import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from compression.huffman.encoder import compress
from compression.huffman.decoder import decompress

logger = logging.getLogger(__name__)

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
    logger.info("Health check requested")
    return {"status": "ok"}


# Compresses input text and returns the encoded result with basic metrics.
@app.post("/compress")
async def compress_endpoint(request: CompressRequest):
    try:
        logger.info("Compression request received")
        logger.info("Original text length: %d characters", len(request.text))

        result = compress(request.text)

        logger.info("Compression completed successfully")
        logger.info("Original bits: %d", result["original_bits"])
        logger.info("Compressed bits: %d", result["compressed_bits"])
        logger.info("Compression ratio: %.4f", result["compression_ratio"])

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
        logger.exception("Compression failed")
        raise HTTPException(status_code=500, detail=str(e))


# Decompresses encoded input data and returns the decoded text.
@app.post("/decompress")
async def decompress_endpoint(request: DecompressRequest):
    try:
        logger.info("Decompression request received")
        logger.info("Encoded data length: %d bits", len(request.encoded_data))

        decoded_text = decompress(request.encoded_data)

        logger.info("Decompression completed successfully")
        logger.info("Decoded text length: %d characters", len(decoded_text))

        return {
            "decoded_text": decoded_text
        }

    # Returns a bad request error if the encoded input is invalid.
    except ValueError as e:
        logger.warning("Invalid encoded input provided for decompression")
        raise HTTPException(status_code=400, detail=str(e))
    # Returns a server error for any other unexpected failure.
    except Exception as e:
        logger.exception("Decompression failed")
        raise HTTPException(status_code=500, detail=str(e))