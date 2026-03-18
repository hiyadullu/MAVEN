import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

labels_path = os.path.join(BASE_DIR, "..", "..", "features", "labels.json")
frames_root = os.path.join(BASE_DIR, "..", "..", "features", "frames")
output_csv = os.path.join(BASE_DIR, "..", "..", "features", "visual_metadata.csv")

with open(labels_path, "r") as f:
    labels = json.load(f)

rows = []

for utt_id, label in labels.items():
    parts = utt_id.split("_")

    # Example:
    # Ses01F_impro01_F000
    # Ses01F_script01_1_M000

    if len(parts) < 3:
        continue

    video_id = "_".join(parts[:-1])  # everything except last
    frame_part = parts[-1]           # F000 or M000

    frame_number = frame_part[1:]    # remove F or M
    frame_index = int(frame_number)  # convert to int

    frame_path = os.path.join(
        frames_root,
        video_id,
        f"{frame_index}.jpg"
    )

    if os.path.exists(frame_path):
        rows.append([frame_path, label])

df = pd.DataFrame(rows, columns=["image_path", "label"])
df.to_csv(output_csv, index=False)

print(f"[+] Saved {len(df)} valid visual samples")
