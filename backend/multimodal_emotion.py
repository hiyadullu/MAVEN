"""
multimodal_emotion.py
Face (YOLO + DeepFace) + Voice emotion (pyAudioAnalysis if available; else heuristic).
Background audio thread -> non-blocking video loop.

Requires:
  pip install ultralytics deepface sounddevice numpy opencv-python scipy librosa pyAudioAnalysis(optional)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPFACE_BACKEND"] = "torch"  # prefer torch backend for DeepFace

import time
import threading
import queue
import traceback

import cv2
import numpy as np
import sounddevice as sd
import torch
from ultralytics import YOLO
from deepface import DeepFace

# ---------- Configuration ----------
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 1.0      # how long each audio chunk is (seconds)
VOICE_UPDATE_INTERVAL = 1.0    # for UI updates (seconds)
VOICE_LABEL_POS = (10, 30)     # where to draw voice label on frame
# -----------------------------------

# ---------- Try to import pyAudioAnalysis (optional) ----------
USE_PYAUDIOANALYSIS = False
try:
    from pyAudioAnalysis import audioTrainTest as aT
    from pyAudioAnalysis import MidTermFeatures as mF
    USE_PYAUDIOANALYSIS = True
    print("pyAudioAnalysis available — will use its pretrained classifier if possible.")
except Exception:
    print("pyAudioAnalysis not available. Will use heuristic fallback for voice emotion.")

# ---------- A safe helper to save temp wav ----------
from scipy.io.wavfile import write as wav_write

def save_temp_wav(filename, audio_np, sr=AUDIO_SAMPLE_RATE):
    # audio_np: 1D float32 np array in range [-1,1] or similar
    # convert to int16 for wav if needed
    arr = np.asarray(audio_np)
    # normalize to int16
    # avoid clipping
    if arr.dtype != np.int16:
        maxv = np.max(np.abs(arr)) if np.max(np.abs(arr)) > 0 else 1.0
        scaled = (arr / maxv * 32767.0).astype(np.int16)
    else:
        scaled = arr
    wav_write(filename, sr, scaled)

# ---------- Voice emotion via pyAudioAnalysis (if available) ----------
def voice_emotion_pyaudio(wav_path):
    """
    Use pyAudioAnalysis file_classification if its pretrained model exists.
    This expects a model name and type. pyAudioAnalysis does not ship a
    guaranteed prepackaged emotion model in pip installs, so this will try
    a common name and gracefully fail to fallback.
    """
    try:
        # Common model names used in forks/examples — adapt if you have a model file.
        # Try 'svmSpeechEmotion' or 'svm_rbf_speech_emotion' as possible names.
        candidates = [
            ("svmSpeechEmotion", "svm"),
            ("svm_rbf_speech_emotion", "svm"),
            ("svm", "svm")
        ]
        for model_name, model_type in candidates:
            try:
                # aT.file_classification returns [class, prob, classNames]
                result, prob, classes = aT.file_classification(wav_path, model_name, model_type)
                idx = int(result)
                return str(classes[idx])
            except Exception:
                continue
        # If none worked, raise so fallback is used
        raise RuntimeError("No usable pyAudioAnalysis model found.")
    except Exception as e:
        # bubble up so caller will use fallback
        raise

# ---------- Simple heuristic fallback (energy + pitch) ----------
import librosa

def voice_emotion_heuristic(wav_np, sr=AUDIO_SAMPLE_RATE):
    """
    Very simple heuristic: use RMS and pitch to guess emotion.
    Not as accurate as ML models but works without extra models.
    Returns one of: 'angry','happy','sad','neutral'
    """
    try:
        y = wav_np.astype(np.float32).flatten()
        if y.size == 0:
            return "unknown"
        # compute rms
        rms = np.mean(librosa.feature.rms(y=y, frame_length=1024, hop_length=512))
        # compute pitch median using librosa.pyin (may return NaNs)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr, frame_length=1024, hop_length=512)
        # median pitch over voiced frames
        if f0 is not None:
            f0_med = np.nanmedian(f0)
            if np.isnan(f0_med):
                f0_med = 0.0
        else:
            f0_med = 0.0
        # zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        # heuristics:
        # high rms + high pitch => angry or happy (distinguish by short-term zcr)
        # low rms + low pitch => sad
        # moderate => neutral
        if rms > 0.04 and f0_med > 160:
            # high energy and pitch — likely happy/angry
            if zcr > 0.15:
                return "angry"
            else:
                return "happy"
        if rms < 0.01 and f0_med < 120:
            return "sad"
        # surprise if sudden high pitch but short duration (hard to detect reliably)
        return "neutral"
    except Exception as e:
        print("heuristic error:", e)
        return "unknown"

# ---------- Background recorder thread ----------
class VoiceWorker(threading.Thread):
    def __init__(self, sr=AUDIO_SAMPLE_RATE, chunk_secs=AUDIO_CHUNK_SECONDS):
        super().__init__(daemon=True)
        self.sr = sr
        self.chunk_secs = chunk_secs
        self.running = True
        self.latest_emotion = "listening..."
        self.q = queue.Queue()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                frames = sd.rec(int(self.sr * self.chunk_secs), samplerate=self.sr, channels=1, dtype='float32')
                sd.wait()
                audio_np = frames.flatten()
                # save temp file (some classifiers expect wav)
                wav_path = "temp_voice_chunk.wav"
                save_temp_wav(wav_path, audio_np, sr=self.sr)

                # first try pyAudioAnalysis if available
                emotion = None
                if USE_PYAUDIOANALYSIS:
                    try:
                        emotion = voice_emotion_pyaudio(wav_path)
                    except Exception:
                        emotion = None

                # fallback heuristic
                if emotion is None:
                    try:
                        emotion = voice_emotion_heuristic(audio_np, sr=self.sr)
                    except Exception:
                        emotion = "unknown"

                self.latest_emotion = emotion
            except Exception as e:
                print("VoiceWorker error:", e)
                traceback.print_exc()
                self.latest_emotion = "unknown"
                time.sleep(0.5)

# ---------- Initialize models ----------
print("Loading YOLO face model (ONNX)...")
# replace with your actual onnx/pytorch model path if needed
YOLO_MODEL_PATH = "yolov8n-face-lindevs.onnx"
face_model = YOLO(YOLO_MODEL_PATH)

print("YOLO model loaded. Initializing camera and DeepFace...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam. Check camera index.")

# Start voice worker thread
voice_worker = VoiceWorker(sr=AUDIO_SAMPLE_RATE, chunk_secs=AUDIO_CHUNK_SECONDS)
voice_worker.start()

# ---------- Main loop ----------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed; exiting.")
            break

        # Face detection using YOLO
        results = face_model(frame)[0]

        face_label = "no-face"
        # process faces (draw first face's emotion)
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # clamp coordinates
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            try:
                # preprocess for DeepFace
                face_resized = cv2.resize(face, (224, 224))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                analysis = DeepFace.analyze(
                    img_path=face_rgb,
                    actions=['emotion'],
                    detector_backend='skip',
                    enforce_detection=False
                )
                # DeepFace may return list or dict
                if isinstance(analysis, list):
                    analysis = analysis[0]
                face_label = analysis.get("dominant_emotion", "unknown")
            except Exception as e:
                # fallback
                print("DeepFace error:", e)
                face_label = "unknown"

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, face_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # only draw first face
            break

        # Get latest voice emotion from worker
        voice_label = voice_worker.latest_emotion

        # Draw voice label
        cv2.putText(frame, f"Voice: {voice_label}", VOICE_LABEL_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

        # Optionally: fuse simple rule (face + voice)
        # Example: if both agree -> final, else voice prioritized if high-energy heuristics, etc.
        # Here we just show both.

        cv2.imshow("Multimodal Emotion (Face + Voice)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Stopping...")
    try:
        voice_worker.stop()
    except:
        pass
    cap.release()
    cv2.destroyAllWindows()
