FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the current directory contents into /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Expose the port the app runs on
EXPOSE 8000

# Define environment variable
ENV PYTHONPATH=/app

# Run the uvicorn server
CMD ["uvicorn", "compression.service.huffmanAPI:app", "--host", "0.0.0.0", "--port", "8000"]
