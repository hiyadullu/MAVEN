# EmotiLearn - All API Endpoints Reference

## 🎯 Quick Reference

### 📄 Page Routes

| Route       | Method | Description              |
| ----------- | ------ | ------------------------ |
| `/`         | GET    | Main landing page        |
| `/face`     | GET    | Live face detection page |
| `/test`     | GET    | Emotion quiz page        |
| `/practice` | GET    | Practice mode page       |
| `/progress` | GET    | Progress tracking page   |
| `/history`  | GET    | Detection history page   |
| `/audio`    | GET    | Audio emotion test UI    |

---

### 👁️ Facial Emotion Detection

#### Start Camera

```
GET /start_camera
```

**Response:** `{"status": "Camera started"}`

#### Stop Camera

```
GET /stop_camera
```

**Response:** `{"status": "Camera stopped"}`

#### Video Stream

```
GET /video_feed
```

**Response:** MJPEG video stream with emotion overlays

#### Get Current Emotions

```
GET /emotions
```

**Response:**

```json
{
  "0": {
    "emotion": "happy",
    "confidence": 0.89,
    "coords": [x1, y1, x2, y2]
  }
}
```

---

### 🎤 Audio Emotion Detection

#### Record & Predict

```
POST /audio/record
Content-Type: application/json

{
  "duration": 7,
  "sample_rate": 22050
}
```

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

#### Predict from File

```
POST /audio/predict_file
Content-Type: multipart/form-data

file: <audio file>
```

**Response:**

```json
{
  "emotion": "sad",
  "confidence": 0.7623,
  "probabilities": {...}
}
```

---

#### Predict from NumPy Array

```
POST /audio/predict_numpy
Content-Type: application/json

{
  "audio": [0.012, 0.034, 0.056, ...],
  "sample_rate": 22050
}
```

**Response:**

```json
{
  "emotion": "neutral",
  "confidence": 0.6234,
  "probabilities": {...}
}
```

---

#### Get Audio Capabilities

```
GET /audio/info
```

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

## 🧪 Testing All Endpoints

### Using curl

#### Test Pages

```bash
curl http://localhost:5000/
curl http://localhost:5000/face
curl http://localhost:5000/audio
```

#### Test Facial Detection

```bash
# Start camera
curl http://localhost:5000/start_camera

# Get emotions
curl http://localhost:5000/emotions

# Get video stream (opens stream)
curl http://localhost:5000/video_feed

# Stop camera
curl http://localhost:5000/stop_camera
```

#### Test Audio Detection

```bash
# Get capabilities
curl http://localhost:5000/audio/info

# Record and predict (7 seconds)
curl -X POST http://localhost:5000/audio/record \
  -H "Content-Type: application/json" \
  -d '{"duration": 7, "sample_rate": 22050}'

# Upload file
curl -X POST http://localhost:5000/audio/predict_file \
  -F "file=@audio.wav"
```

### Using Python

```python
import requests

# Audio info
response = requests.get('http://localhost:5000/audio/info')
print(response.json())

# Record audio
response = requests.post('http://localhost:5000/audio/record',
    json={'duration': 7, 'sample_rate': 22050})
print(response.json())

# Upload file
with open('audio.wav', 'rb') as f:
    response = requests.post('http://localhost:5000/audio/predict_file',
        files={'file': f})
    print(response.json())
```

### Using JavaScript

```javascript
// Get capabilities
fetch("/audio/info")
    .then((r) => r.json())
    .then((data) => console.log(data));

// Record audio
fetch("/audio/record", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ duration: 7, sample_rate: 22050 }),
})
    .then((r) => r.json())
    .then((data) => console.log(data));

// Upload file
const formData = new FormData();
formData.append("file", fileInput.files[0]);
fetch("/audio/predict_file", {
    method: "POST",
    body: formData,
})
    .then((r) => r.json())
    .then((data) => console.log(data));
```

---

## 📊 Response Codes

| Code | Meaning      | Example                       |
| ---- | ------------ | ----------------------------- |
| 200  | Success      | Emotion detected successfully |
| 400  | Bad Request  | Missing required parameter    |
| 500  | Server Error | Model initialization failed   |

---

## 🎨 Emotion Mappings

```javascript
const emotionIcons = {
    angry: "😠",
    disgust: "🤮",
    fear: "😨",
    happy: "😊",
    neutral: "😐",
    sad: "😢",
    surprise: "😮",
};
```

---

## ⚙️ Configuration Parameters

### Audio Recording

-   **Duration:** 1-30 seconds (default: 7)
-   **Sample Rate:** Any supported rate (default: 22050 Hz)
-   **Format:** WAV/MP3/M4A/FLAC (auto-detected)

### Feature Extraction

-   **MFCC Coefficients:** 13
-   **Chroma Features:** 12
-   **Total Features:** 26

### Model Settings

-   **Confidence Threshold:** 0.5 (50%)
-   **Algorithm:** Support Vector Machine (SVM)
-   **Supported Emotions:** 7

---

## 🔄 Full Workflow Example

```python
import requests
import json

BASE_URL = 'http://localhost:5000'

# 1. Check capabilities
print("🔍 Checking audio capabilities...")
response = requests.get(f'{BASE_URL}/audio/info')
info = response.json()
print(f"Supported emotions: {info['emotions']}")
print(f"Recommended duration: {info['duration_recommended']} seconds")

# 2. Record and predict
print("\n🎤 Recording audio...")
response = requests.post(f'{BASE_URL}/audio/record',
    json={'duration': 7, 'sample_rate': 22050})
result = response.json()

# 3. Display results
print(f"\n📊 Results:")
print(f"Detected Emotion: {result['emotion'].upper()}")
print(f"Confidence: {result['confidence']*100:.1f}%")

# 4. Show probabilities
print(f"\nProbability breakdown:")
for emotion, prob in result['probabilities'].items():
    print(f"  {emotion:10s}: {prob*100:5.1f}%")
```

**Output:**

```
🔍 Checking audio capabilities...
Supported emotions: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
Recommended duration: 7 seconds

🎤 Recording audio...

📊 Results:
Detected Emotion: HAPPY
Confidence: 85.2%

Probability breakdown:
  angry     :  2.3%
  disgust   :  1.6%
  fear      :  3.5%
  happy     : 85.2%
  neutral   :  5.4%
  sad       :  1.2%
  surprise  :  0.8%
```

---

## 🚀 Deployment Endpoints

Once deployed, replace `localhost:5000` with your domain:

```
Web:   https://yourdomain.com/audio
API:   https://yourdomain.com/audio/record
       https://yourdomain.com/audio/predict_file
       https://yourdomain.com/audio/info
```

---

## 📝 Notes

-   All timestamps in UTC
-   Confidence scores: 0.0 to 1.0 (multiply by 100 for percentage)
-   Audio is automatically preprocessed (silence trimming, normalization)
-   Maximum file size for upload: Depends on Flask configuration (default ~16MB)
-   Emotions below 50% confidence default to "neutral"

---

## 🔗 Related Documentation

-   **Full API Guide:** See `AUDIO_EMOTION_API.md`
-   **Integration Guide:** See `INTEGRATION_GUIDE.md`
-   **Implementation Status:** See `INTEGRATION_COMPLETE.md`
