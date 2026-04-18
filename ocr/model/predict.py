"""
predict.py — Run the trained OCR CNN on a single image.

Usage:
    python predict.py --image path/to/digit.png
    python predict.py --image path/to/digit.png --weights best_model.pth

The input image should be:
  - Grayscale (or will be converted automatically)
  - Any size (will be resized to 28x28)
  - White digit on black background, or black digit on white (auto-inverted if needed)
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from ocr.model.cnn import OCRCNN

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081


def load_image(path: str) -> torch.Tensor:
    """
    Load any image from disk and prepare it as a (1, 1, 28, 28) tensor
    matching the MNIST preprocessing pipeline.
    """
    img = Image.open(path).convert("L")   # force grayscale

    # MNIST digits are white-on-black. If the image looks inverted, flip it.
    # Heuristic: if the average pixel value > 127, the background is bright.
    import numpy as np
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    return transform(img).unsqueeze(0)   # (1, 1, 28, 28)


def predict(image_path: str, weights_path: str = "best_model.pth") -> dict:
    """
    Run inference on a single image.

    Returns:
        dict with keys:
            'prediction'   — predicted digit (int)
            'confidence'   — softmax probability of the predicted class (float)
            'probabilities'— list of probabilities for all 10 classes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = OCRCNN(num_classes=10).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load and preprocess image
    tensor = load_image(image_path).to(device)

    with torch.no_grad():
        logits = model(tensor)                        # (1, 10)
        probs  = F.softmax(logits, dim=1).squeeze()  # (10,)

    pred       = probs.argmax().item()
    confidence = probs[pred].item()
    all_probs  = probs.tolist()

    return {
        "prediction":    pred,
        "confidence":    confidence,
        "probabilities": all_probs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict digit from image")
    parser.add_argument("--image",   required=True, help="Path to input image")
    parser.add_argument("--weights", default="best_model.pth", help="Path to model weights")
    args = parser.parse_args()

    result = predict(args.image, args.weights)

    print(f"Predicted digit : {result['prediction']}")
    print(f"Confidence      : {result['confidence']:.2%}")
    print("\nAll class probabilities:")
    for digit, prob in enumerate(result["probabilities"]):
        bar = "█" * int(prob * 40)
        print(f"  {digit}: {prob:6.2%}  {bar}")