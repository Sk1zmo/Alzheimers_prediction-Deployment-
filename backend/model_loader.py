import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Import model architecture
from model_architecture import AlzheimerCNN

# Storage + Database
from storage import upload_image_to_supabase
from database import save_prediction


# ============================================
# DEVICE + MODEL PATH
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "alzheimers_model.pth")

# Classes (adjust if needed)
CLASS_NAMES = ["Class0", "Class1", "Class2", "Class3"]


# ============================================
# PREPROCESSING PIPELINE
# ============================================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ============================================
# LOAD MODEL
# ============================================
def load_model():
    print("Loading model from:", MODEL_PATH)

    model = AlzheimerCNN(num_classes=4)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model.to(DEVICE)


model = load_model()


# ============================================
# (1) CT SCAN DETECTOR — SIMPLE HEURISTIC
# ============================================
def is_ct_scan(image: Image.Image):
    arr = np.array(image.convert("L"))
    std = arr.std()
    mean = arr.mean()

    if std < 20 or std > 140:
        return False
    if mean < 40 or mean > 200:
        return False

    return True


# ============================================
# (2) GRAD-CAM GENERATION
# ============================================
def generate_gradcam(model, image_tensor):
    model.eval()

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook last conv layer
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


# ============================================
# Optional ROC Curve Calculation
# ============================================
def compute_simple_roc(probabilities, predicted_class):
    p = float(probabilities[predicted_class])
    thresholds = np.linspace(0, 1, 20)

    fpr = [float(abs(t - p)) for t in thresholds]
    tpr = [float(max(0, 1 - abs(t - p))) for t in thresholds]

    return list(fpr), list(tpr), list(thresholds)


# ============================================
# (3) MAIN PREDICTION FUNCTION (SUPABASE + DB)
# ============================================
def predict_image(image: Image.Image, filename: str, image_bytes: bytes):

    # -------------------------------
    # Preprocessing
    # -------------------------------
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    predicted_class = int(np.argmax(probabilities))
    confidence_value = float(probabilities[predicted_class])

    # -------------------------------
    # Extra Metrics (same as before)
    # -------------------------------
    f1_est = (2 * confidence_value * confidence_value) / (
        confidence_value + confidence_value + 1e-9
    )

    one_hot = np.zeros(len(probabilities))
    one_hot[predicted_class] = 1
    mse = float(np.mean((probabilities - one_hot) ** 2))

    fpr, tpr, thresholds = compute_simple_roc(probabilities, predicted_class)

    # -------------------------------
    # Upload to Supabase
    # -------------------------------
    image_url = upload_image_to_supabase(filename, image_bytes)

    # -------------------------------
    # Save metadata into SQLite DB
    # -------------------------------
    save_prediction(filename, image_url, CLASS_NAMES[predicted_class], confidence_value)

    # -------------------------------
    # Final response
    # -------------------------------
    return {
        "class_id": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": confidence_value,
        "probabilities": probabilities.tolist(),
        "image_url": image_url,
        "is_ct_scan": is_ct_scan(image),

        "metrics": {
            "f1_estimate": float(f1_est),
            "mse": mse
        },

        "roc": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds
        }
    }


# ============================================
# (4) GRAD-CAM ENDPOINT SUPPORT
# ============================================
def explain_image(image: Image.Image):
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    heatmap = generate_gradcam(model, img_tensor)

    success, buffer = cv2.imencode(".png", heatmap)
    return buffer.tobytes()
