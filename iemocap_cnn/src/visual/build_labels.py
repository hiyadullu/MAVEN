import os
import re
import json

# Adjust base path relative to THIS file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IEMOCAP_PATH = os.path.join(
    BASE_DIR, "..", "..", "data", "IEMOCAP_full_release"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "..", "..", "features", "labels.json"
)

VALID_EMOTIONS = {
    "ang": 0,
    "hap": 1,
    "exc": 1,   # merge excited into happy
    "sad": 2,
    "neu": 3
}

labels = {}

for session in sorted(os.listdir(IEMOCAP_PATH)):
    if not session.startswith("Session"):
        continue

    session_path = os.path.join(IEMOCAP_PATH, session)
    eval_dir = os.path.join(session_path, "dialog", "EmoEvaluation")

    if not os.path.exists(eval_dir):
        continue

    print(f"[INFO] Processing {session}")

    for file in os.listdir(eval_dir):
        if not file.endswith(".txt"):
            continue

        with open(os.path.join(eval_dir, file), "r") as f:
            lines = f.readlines()

        for line in lines:
            match = re.match(
                r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)\s+\[",
                line
            )

            if match:
                utt_id = match.group(3)
                emotion = match.group(4).lower()

                if emotion in VALID_EMOTIONS:
                    labels[utt_id] = VALID_EMOTIONS[emotion]

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(labels, f)

print("\nSaved labels:", len(labels))
