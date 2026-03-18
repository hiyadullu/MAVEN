import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# PATHS (LOCKED)
# =========================
DATASET_PATH = os.path.join(
    BASE_DIR, "..", "..", "data", "IEMOCAP_full_release"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "..", "..", "features", "frames"
)

CASCADE_PATH = os.path.join(
    BASE_DIR, "haarcascade_frontalface_default.xml"
)

os.makedirs(OUTPUT_PATH, exist_ok=True)


face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError(f"Failed to load Haar Cascade from {CASCADE_PATH}")


# =========================
# FACE EXTRACTION
# =========================
def extract_faces(video_path, save_dir):

    # ✅ RESUME LOGIC
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 5:
        print("     Skipping (already processed)")
        return

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces[:1]:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))

            cv2.imwrite(
                os.path.join(save_dir, f"{frame_id}.jpg"),
                face
            )

            frame_id += 1
            break

    cap.release()

# =========================
# MAIN LOOP
# =========================
def main():
    for session in sorted(os.listdir(DATASET_PATH)):
        if not session.startswith("Session"):
            continue

        session_path = os.path.join(DATASET_PATH, session)
        video_root = os.path.join(session_path, "dialog", "avi", "DivX")

        if not os.path.exists(video_root):
            continue

        print(f"[INFO] Processing {session}")

        for video in os.listdir(video_root):
            if not video.endswith(".avi"):
                continue

            utt_id = video.replace(".avi", "")
            video_path = os.path.join(video_root, video)
            save_dir = os.path.join(OUTPUT_PATH, utt_id)

            print(f"  → {utt_id}")
            extract_faces(video_path, save_dir)


if __name__ == "__main__":
    main()
