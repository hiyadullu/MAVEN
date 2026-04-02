import os
import shutil
import random

# -------- CONFIG --------
SOURCE_DIR = "data/processed/video_data"
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
SPLIT_RATIO = 0.8  # 80% train, 20% val
SEED = 42

random.seed(SEED)

# -------- CREATE FOLDERS --------
for split in [TRAIN_DIR, VAL_DIR]:
    for cls in ['0', '1', '2', '3']:
        os.makedirs(os.path.join(split, cls), exist_ok=True)

# -------- SPLIT LOGIC --------
for cls in ['0', '1', '2', '3']:
    class_path = os.path.join(SOURCE_DIR, cls)
    videos = os.listdir(class_path)

    random.shuffle(videos)

    split_idx = int(len(videos) * SPLIT_RATIO)

    train_videos = videos[:split_idx]
    val_videos = videos[split_idx:]

    # Move folders
    for vid in train_videos:
        src = os.path.join(class_path, vid)
        dst = os.path.join(TRAIN_DIR, cls, vid)
        shutil.move(src, dst)

    for vid in val_videos:
        src = os.path.join(class_path, vid)
        dst = os.path.join(VAL_DIR, cls, vid)
        shutil.move(src, dst)

print("✅ Dataset split into train/val successfully!")