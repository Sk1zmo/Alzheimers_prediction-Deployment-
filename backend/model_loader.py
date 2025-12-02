import os
import torch
from torchvision import transforms
from PIL import Image

from model_architecture import AlzheimerCNN   # Must match your training model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "alzheimers_model.pth")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    print("Loading model from:", MODEL_PATH)

    # Create empty model (architecture)
    model = AlzheimerCNN(num_classes=4)

    # Load weights (state_dict)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # Apply weights
    model.load_state_dict(state_dict)

    model.eval()
    return model.to(DEVICE)

# Load on startup
model = load_model()

def predict_image(image: Image.Image):
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)

    return {
        "class_id": int(predicted.item()),
        "confidence": float(confidence.item()),
        "probabilities": probabilities.tolist()
    }
