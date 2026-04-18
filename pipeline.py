import requests
import sys
import os

def run_pipeline(image_path, api_url="http://localhost:8080/pipeline"):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        return

    print(f"Sending image {image_path} to pipeline...")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            response = requests.post(api_url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("\n--- Pipeline Result ---")
            print(f"OCR Prediction    : {result['ocr_result']['prediction']}")
            print(f"OCR Confidence    : {result['ocr_result']['confidence']:.2%}")
            print(f"Original Text     : {result['decompressed_text']}")
            print(f"Encoded Bits      : {result['compression_result']['encoded_bits']}")
            print(f"Compression Ratio : {result['compression_result']['ratio']:.4f}")
            print("------------------------")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure the FastAPI app is running (e.g., 'python app.py' or in Docker).")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "test_digit.png"
        print(f"No image path provided. Usage: python test_pipeline.py <path_to_image>")
        
    run_pipeline(img_path)
