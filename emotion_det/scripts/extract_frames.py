import os
import cv2
from tqdm import tqdm

# =====================
# PATHS (EDIT ONLY IF NEEDED)
# =====================
IEMOCAP_DIR = "data/raw/iemocap/videos"
RAVDESS_DIR = "data/raw/ravdess/videos"

OUTPUT_DIR = "data/processed/video_data"

FRAME_COUNT = 8

# =====================
# LABEL MAPPING
# =====================

# RAVDESS mapping
def get_ravdess_label(filename):
    try:
        code = int(filename.split("-")[2])
    except:
        return None

    if code == 5:
        return 0  # Angry
    elif code == 3:
        return 1  # Happy
    elif code == 4:
        return 2  # Sad
    elif code in [1, 2]:
        return 3  # Neutral
    else:
        return None  # drop unwanted emotions


# IEMOCAP mapping (heuristic)
def get_iemocap_label(filename):
    fname = filename.lower()

    if "ang" in fname:
        return 0
    elif "hap" in fname or "exc" in fname:
        return 1
    elif "sad" in fname:
        return 2
    elif "neu" in fname:
        return 3
    else:
        return None


# =====================
# FRAME EXTRACTION
# =====================
def extract_frames(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < FRAME_COUNT:
        cap.release()
        return

    interval = total_frames // FRAME_COUNT

    count = 0
    saved = 0

    while cap.isOpened() and saved < FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(os.path.join(save_dir, f"frame_{saved}.jpg"), frame)
            saved += 1

        count += 1

    cap.release()


# =====================
# PROCESS FUNCTION
# =====================
def process_dataset(video_dir, label_func):
    for file in tqdm(os.listdir(video_dir)):
        if not file.endswith((".mp4", ".avi")):
            continue

        label = label_func(file)
        if label is None:
            continue

        video_path = os.path.join(video_dir, file)
        video_name = file.split(".")[0]

        save_dir = os.path.join(OUTPUT_DIR, str(label), video_name)
        os.makedirs(save_dir, exist_ok=True)

        extract_frames(video_path, save_dir)


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    print("Processing IEMOCAP...")
    process_dataset(IEMOCAP_DIR, get_iemocap_label)

    print("Processing RAVDESS...")
    process_dataset(RAVDESS_DIR, get_ravdess_label)

    print("Done.")