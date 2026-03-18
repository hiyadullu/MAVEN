import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from src.visual.model import get_model

# ======================
# SETTINGS
# ======================
MODEL_PATH = "model.pth"
NUM_CLASSES = 4
device = torch.device("cpu")

class_names = ["Angry", "Happy", "Neutral", "Sad"]

# ======================
# LOAD MODEL
# ======================
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ======================
# TRANSFORM (FIXED)
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45],
                         std=[0.225, 0.225, 0.225])
])

# ======================
# FACE DETECTOR
# ======================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ======================
# WEBCAM LOOP
# ======================
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        pil_image = Image.fromarray(face_rgb)

        # Apply transform
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        label = f"{class_names[predicted.item()]} ({confidence.item()*100:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

