import torch
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_path="alzheimers_model.pth"):
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()
    return model.to(DEVICE)

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
