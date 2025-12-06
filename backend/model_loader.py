import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from model_architecture import AlzheimerCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "alzheimers_model.pth")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ------------------------
# Load Model
# ------------------------
def load_model():
    print("Loading model from:", MODEL_PATH)
    model = AlzheimerCNN(num_classes=4)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE)

model = load_model()


# ------------------------
# CT Detector
# ------------------------
def is_ct_scan(image: Image.Image):
    arr = np.array(image.convert("L"))
    std = arr.std()
    mean = arr.mean()

    if std < 20 or std > 140: return False
    if mean < 40 or mean > 200: return False
    return True


# ------------------------
# Grad-CAM Generator
# ------------------------
def generate_gradcam(model, image_tensor):
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, __, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.conv3
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[-1]
    acts = activations[-1]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze().detach().cpu().numpy()

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-9)

    cam = cv2.resize(cam, (224, 224))
    heatmap = (cam * 255).astype(np.uint8)

    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


# ------------------------
# Main Prediction
# ------------------------
def predict_image(image: Image.Image):
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    class_id = int(torch.argmax(probs))
    confidence = float(probs[class_id])

    # -----------------------------
    # Metrics
    # -----------------------------
    # F1 estimate
    f1_est = (2 * confidence * confidence) / (confidence + confidence + 1e-9)

    # MSE & MSME
    one_hot = np.zeros(4)
    one_hot[class_id] = 1

    mse = float(np.mean((probs.cpu().numpy() - one_hot) ** 2))
    msme = mse * confidence

    # -----------------------------
    # ROC curve
    # -----------------------------
    y_true = one_hot
    y_scores = probs.cpu().numpy()

    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
    except:
        fpr, tpr, roc_auc = [0], [0], 0.0

    return {
        "class_id": class_id,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "is_ct_scan": is_ct_scan(image),
        "metrics": {
            "f1": float(f1_est),
            "mse": mse,
            "msme": msme,
            "roc": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc)
            }
        }
    }


# ------------------------
# Explanation & Heatmap
# ------------------------
def explain_image(image: Image.Image):
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    heatmap = generate_gradcam(model, img_tensor)
    ok, buffer = cv2.imencode(".png", heatmap)
    return buffer.tobytes()

