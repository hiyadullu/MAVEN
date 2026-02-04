# EmotiLearn - Audio + Video Emotion Detection API

This document describes the integrated emotion detection API that combines both visual (facial) and audio emotion detection.

## Overview

The backend now supports:

-   **Video/Facial Emotion Detection** - Using DeepFace for real-time face emotion analysis
-   **Audio Emotion Detection** - Using pre-trained SVM model trained on speech emotion data

## Setup

### Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. The trained audio models are already included:

    - `svm_model.pkl` - Pre-trained SVM classifier
    - `scaler.pkl` - Feature scaler for normalization
    - `label_encoder.pkl` - Emotion label encoder

3. Run the Flask app:

```bash
python app.py
```

---

## API Endpoints

### FACIAL EMOTION DETECTION

#### 1. Start Camera

**Endpoint:** `GET /start_camera`

Starts the camera feed for real-time facial emotion detection.

**Response:**

```json
{
    "status": "Camera started"
}
```

---

#### 2. Stop Camera

**Endpoint:** `GET /stop_camera`

Stops the camera feed.

**Response:**

```json
{
    "status": "Camera stopped"
}
```

---

#### 3. Video Feed Stream

**Endpoint:** `GET /video_feed`

Returns an MJPEG stream of video frames with emotion detection overlays.

**Response:** Video stream with:

-   Bounding boxes around detected faces
-   Emotion label and confidence score
-   Color-coded boxes (green for high confidence, orange for low)

---

#### 4. Get Current Emotions

**Endpoint:** `GET /emotions`

Returns the emotions currently detected in the camera feed.

**Response:**

```json
{
  "0": {
    "emotion": "happy",
    "confidence": 0.89,
    "coords": [x1, y1, x2, y2]
  },
  "1": {
    "emotion": "sad",
    "confidence": 0.76,
    "coords": [x1, y1, x2, y2]
  }
}
```

---

### AUDIO EMOTION DETECTION

#### 1. Record & Predict Audio

**Endpoint:** `POST /audio/record`

Records audio from the microphone and detects emotion in real-time.

**Request Body:**

```json
{
    "duration": 7,
    "sample_rate": 22050
}
```

**Parameters:**

-   `duration` (int, optional): Recording duration in seconds (default: 7)
-   `sample_rate` (int, optional): Sample rate in Hz (default: 22050)

**Response:**

```json
{
    "emotion": "happy",
    "confidence": 0.8523,
    "probabilities": {
        "angry": 0.0234,
        "disgust": 0.0156,
        "fear": 0.0345,
        "happy": 0.8523,
        "neutral": 0.0542,
        "sad": 0.0123,
        "surprise": 0.0077
    }
}
```

---

#### 2. Predict from Audio File

**Endpoint:** `POST /audio/predict_file`

Upload an audio file and detect emotion.

**Request:**

-   Content-Type: `multipart/form-data`
-   Form field: `file` (audio file in .wav, .mp3, or other formats supported by librosa)

**Response:**

```json
{
    "emotion": "sad",
    "confidence": 0.7623,
    "probabilities": {
        "angry": 0.0812,
        "disgust": 0.0234,
        "fear": 0.0567,
        "happy": 0.0234,
        "neutral": 0.0434,
        "sad": 0.7623,
        "surprise": 0.0096
    }
}
```

---

#### 3. Predict from Audio Array

**Endpoint:** `POST /audio/predict_numpy`

Send raw audio samples as a JSON array for emotion detection.

**Request Body:**

```json
{
  "audio": [0.012, 0.034, 0.056, ...],
  "sample_rate": 22050
}
```

**Parameters:**

-   `audio` (array): Array of audio samples (required)
-   `sample_rate` (int, optional): Sample rate in Hz (default: 22050)

**Response:** Same as record endpoint

---

#### 4. Audio Info & Capabilities

**Endpoint:** `GET /audio/info`

Returns information about audio emotion detection capabilities.

**Response:**

```json
{
    "status": "ready",
    "emotions": [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"
    ],
    "sample_rate": 22050,
    "feature_count": 26,
    "duration_recommended": 7
}
```

---

## Supported Emotions

Both facial and audio detection support the same emotion classes:

-   **angry** - Anger or frustration
-   **disgust** - Disgust or revulsion
-   **fear** - Fear or anxiety
-   **happy** - Happiness or joy
-   **neutral** - Neutral emotion
-   **sad** - Sadness or sorrow
-   **surprise** - Surprise or shock

---

## Page Routes

### Web Pages

-   `GET /` - Main landing page
-   `GET /face` - Live face detection page
-   `GET /test` - Emotion quiz page
-   `GET /practice` - Practice mode page
-   `GET /progress` - Progress tracking page
-   `GET /history` - Detection history page

---

## Audio Processing Details

### Feature Extraction

The audio emotion detection uses the following features:

1. **MFCC** (Mel-Frequency Cepstral Coefficients) - 13 coefficients
2. **Chroma Features** - 12 chroma features
3. **Zero Crossing Rate** - 1 feature

**Total: 26 features per audio sample**

### Preprocessing

-   **Duration:** 7 seconds recommended
-   **Sample Rate:** 22050 Hz
-   **Silence Trimming:** Automatically removes silence (top_db=25)
-   **Normalization:** All audio is normalized to [-1, 1] range
-   **Scaling:** Features are scaled using StandardScaler

### Model

-   **Algorithm:** Support Vector Machine (SVM)
-   **Training Data:** SAVEE dataset (audio speeches with emotion labels)
-   **Confidence Threshold:** 0.5 (below this, classified as "neutral")

---

## Usage Examples

### Using curl

#### Record audio and detect emotion:

```bash
curl -X POST http://localhost:5000/audio/record \
  -H "Content-Type: application/json" \
  -d '{"duration": 7, "sample_rate": 22050}'
```

#### Upload audio file:

```bash
curl -X POST http://localhost:5000/audio/predict_file \
  -F "file=@path/to/audio.wav"
```

#### Get audio capabilities:

```bash
curl http://localhost:5000/audio/info
```

### Using Python

```python
import requests
import json

# Record and predict
response = requests.post('http://localhost:5000/audio/record',
    json={'duration': 7, 'sample_rate': 22050})
print(response.json())

# Upload file
with open('audio.wav', 'rb') as f:
    response = requests.post('http://localhost:5000/audio/predict_file',
        files={'file': f})
    print(response.json())

# Get capabilities
response = requests.get('http://localhost:5000/audio/info')
print(response.json())
```

---

## File Structure

```
back/
├── app.py                          # Flask application with all routes
├── audio_emotion_detector.py       # Audio emotion detection module
├── audio_feature_extraction.py     # Audio feature extraction functions
├── emotion_cam.py                  # Video emotion detection helper
├── svm_model.pkl                   # Trained SVM model
├── scaler.pkl                      # Feature scaler
├── label_encoder.pkl               # Label encoder
├── requirements.txt                # Python dependencies
├── static/                         # CSS and JS files
│   ├── face.css, history.css, etc.
│   └── script.js, history.js, etc.
└── templates/                      # HTML templates
    ├── face.html, history.html, etc.
    └── index.html
```

---

## Error Handling

All endpoints return appropriate HTTP status codes:

-   **200 OK** - Successful prediction
-   **400 Bad Request** - Invalid input or missing required fields
-   **500 Internal Server Error** - Model initialization or prediction error

Error responses include an `error` field with description:

```json
{
    "error": "Audio detector not initialized",
    "emotion": "neutral",
    "confidence": 0.0
}
```

---

## Performance Notes

### Facial Detection

-   Real-time processing at ~30 FPS (depends on hardware)
-   Requires webcam access
-   GPU recommended for better performance

### Audio Detection

-   Processing time: ~1-2 seconds for 7-second audio
-   Works on CPU (pre-trained, no training required)
-   Typical confidence range: 0.5-0.95

---

## Integration with Frontend

The frontend can integrate with audio detection via:

```javascript
// Record and predict
async function recordAudio() {
    const response = await fetch("/audio/record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ duration: 7, sample_rate: 22050 }),
    });
    return await response.json();
}

// Upload file
async function uploadAudioFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch("/audio/predict_file", {
        method: "POST",
        body: formData,
    });
    return await response.json();
}
```

---

## Future Enhancements

-   Real-time audio streaming for live emotion detection
-   Emotion history database integration
-   Combined facial + audio emotion fusion
-   Mobile app integration
-   WebSocket support for bidirectional communication
