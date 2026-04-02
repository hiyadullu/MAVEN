import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from collections import deque
from transformers import SwinForImageClassification, SwinConfig

# --- 1. MODEL DEFINITION ---
class VideoSwinModel(nn.Module):
    def __init__(self, num_classes=4):
        super(VideoSwinModel, self).__init__()
        config = SwinConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.swin = SwinForImageClassification(config)
        self.swin.classifier = nn.Identity() 
        self.temporal_fc = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.view(-1, c, h, w) 
        features = self.swin.swin(x).last_hidden_state 
        pooled = features.mean(dim=1) 
        video_features = pooled.view(b, f, -1).mean(dim=1)
        return self.temporal_fc(video_features)

# --- 2. THE LOADER ---
def load_my_model(path, device):
    model = VideoSwinModel(num_classes=4).to(device)
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {k.replace('backbone.', '').replace('temporal_head', 'temporal_fc'): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# --- 3. CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABELS = ['Angry', 'Happy', 'Sad', 'Neutral']
MODEL_PATH = 'checkpoints/foundation_model.pth' # Switch to foundation_model.pth to compare

model = load_my_model(MODEL_PATH, DEVICE)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 4. WEBCAM LOOP ---
def run_demo():
    cap = cv2.VideoCapture(0)
    frame_queue = deque(maxlen=8)
    print(f"Running {MODEL_PATH}... Look directly at the camera.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detect Face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        display_label = "Waiting for face..."

        for (x, y, w, h) in faces:
            # 2. Crop and Preprocess
            face_img = frame[y:y+h, x:x+w]
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((224, 224))
            img_tensor = torch.tensor(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
            frame_queue.append(img_tensor)
            
            # 3. Predict
            if len(frame_queue) == 8:
                input_tensor = torch.stack(list(frame_queue)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(input_tensor)
                    # Apply a small "Neutral" threshold - if no emotion is strong, call it Neutral
                    probs = torch.nn.functional.softmax(output, dim=1)
                    if probs.max() < 0.4: # Adjust this sensitivity
                        display_label = "Neutral (Low Confidence)"
                    else:
                        display_label = LABELS[torch.argmax(output, dim=1).item()]
            
            # Draw UI
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, display_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow('MAVEN Face-Only Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

run_demo()