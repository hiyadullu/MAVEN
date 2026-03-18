import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPFACE_BACKEND"] = "torch"

from ultralytics import YOLO
import cv2
from deepface import DeepFace
import numpy as np



face_model = YOLO("yolov8n-face-lindevs.onnx")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = face_model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        try:
            # preprocess typshi
            face_resized = cv2.resize(face, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            analysis = DeepFace.analyze(
                img_path=face_rgb,
                actions=['emotion'],
                detector_backend='skip',
                enforce_detection=False
            )

            # list to dict
            emotion = analysis[0]["dominant_emotion"]

        except Exception as e:
            print("DeepFace error:", e)
            emotion = "unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, emotion, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


