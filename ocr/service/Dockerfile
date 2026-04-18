FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Install system dependencies required for Pillow and potentially PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into /app
COPY . /app

# Install any needed packages
# This will include FastAPI, Uvicorn, Pillow, and PyTorch (CPU version)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    Pillow \
    torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    numpy

# Expose the port the app runs on
EXPOSE 8001

# Define environment variable
ENV PYTHONPATH=/app

# Run the uvicorn server
CMD ["uvicorn", "ocr.service.modelAPI:app", "--host", "0.0.0.0", "--port", "8001"]
