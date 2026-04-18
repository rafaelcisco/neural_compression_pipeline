import torch
from ocr.model.cnn import OCRCNN
import os

def check_model():
    weights_path = "ocr/model/best_model.pth"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found")
        return

    try:
        device = torch.device("cpu")
        model = OCRCNN(num_classes=10).to(device)
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully!")
        
        # Dummy input
        dummy_input = torch.randn(1, 1, 28, 28)
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_model()
