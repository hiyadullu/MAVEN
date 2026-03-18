import os
import librosa
import numpy as np
from utils import parse_emotion

DATA_DIR = "data/IEMOCAP_full_release"
SAVE_DIR = "features/mel_specs"

SAMPLE_RATE = 16000
N_MELS = 128
MAX_LEN = 300

os.makedirs(SAVE_DIR, exist_ok=True)

def extract_mel(wav_path):
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    return librosa.power_to_db(mel)

def pad(spec):
    if spec.shape[1] >= MAX_LEN:
        return spec[:, :MAX_LEN]
    return np.pad(spec, ((0, 0), (0, MAX_LEN - spec.shape[1])))

def process_session(session_path):
    emo_path = os.path.join(session_path, "dialog/EmoEvaluation")
    wav_root = os.path.join(session_path, "dialog/wav")

    for emo_file in os.listdir(emo_path):
        if not emo_file.endswith(".txt"):
            continue

        with open(os.path.join(emo_path, emo_file)) as f:
            for line in f:
                if not line.startswith("["):
                    continue

                label = parse_emotion(line)
                if label is None:
                    continue

                utt_id = line.split("\t")[1]
                wav_path = os.path.join(wav_root, emo_file[:-4], utt_id + ".wav")

                if not os.path.exists(wav_path):
                    continue

                mel = pad(extract_mel(wav_path))
                np.save(os.path.join(SAVE_DIR, f"{utt_id}_{label}.npy"), mel)

def main():
    for s in os.listdir(DATA_DIR):
        if s.startswith("Session"):
            print("Processing", s)
            process_session(os.path.join(DATA_DIR, s))

if __name__ == "__main__":
    main()
