import sounddevice as sd
import numpy as np
import onnxruntime as ort
from scipy.signal import resample

# -----------------------------------
# Load ONNX Model
# -----------------------------------
session = ort.InferenceSession("emotion.onnx", providers=["CPUExecutionProvider"])

# Load emotion labels
with open("labels.txt", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

SAMPLE_RATE = 16000
DURATION = 1  # seconds


def get_audio_emotion():
    try:
        # Record audio
        print("🎤 Listening...")
        audio = sd.rec(int(SAMPLE_RATE * DURATION),
                       samplerate=SAMPLE_RATE,
                       channels=1)
        sd.wait()

        audio = audio.flatten()

        # Normalize audio
        audio = audio / np.max(np.abs(audio))

        # ONNX expects shape (1, audio_length)
        audio_input = audio.astype(np.float32).reshape(1, -1)

        # Run ONNX inference
        outputs = session.run(None, {"input_values": audio_input})
        scores = outputs[0][0]

        # Pick highest score
        emotion = CLASSES[np.argmax(scores)]

        return emotion

    except Exception as e:
        print("Audio Error:", e)
        return "unknown"


# -----------------------------------
# Live loop
# -----------------------------------
print("🎧 Real-Time Audio Emotion Recognition")
print("Press CTRL+C to stop.\n")

while True:
    emotion = get_audio_emotion()
    print("Detected Emotion:", emotion)
