import librosa
import numpy as np

# =========================================
# For DATASET (wav files)
# =========================================
def extract_features(file_path, sr=22050):
    """Extract features from audio file"""
    y, sr = librosa.load(file_path, sr=sr)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # Normalize
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([mfcc_mean, chroma_mean, zcr])


# =========================================
# For REAL-TIME microphone audio
# =========================================
def extract_features_from_audio(y, sr):
    """Extract features from real-time audio samples"""
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # Normalize
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([mfcc_mean, chroma_mean, zcr])
