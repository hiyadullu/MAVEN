import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPFACE_BACKEND"] = "torch"

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
from deepface import DeepFace
import numpy as np
import threading
import sounddevice as sd
import librosa
from audio_emotion_detector import get_detector
import logging
import json

# Custom JSON Encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, (np.str_, str)):
            return str(obj)
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use DeepFace's built-in face detection (more accurate for emotion detection)
# We'll use DeepFace.extract_faces which handles detection internally

# Global variables for camera and threading
camera = None
camera_active = False
current_emotions = {}
lock = threading.Lock()


def get_frame():
    """Generator function to stream video frames with emotion detection"""
    global camera_active, current_emotions
    
    cap = cv2.VideoCapture(0)
    
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Use DeepFace to extract faces and analyze emotions
            faces = DeepFace.extract_faces(
                img_path=frame,
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            with lock:
                current_emotions.clear()
            
            for idx, face_data in enumerate(faces):

                x = int(face_data['facial_area']['x'])
                y = int(face_data['facial_area']['y'])
                w = int(face_data['facial_area']['w'])
                h = int(face_data['facial_area']['h'])
                
                x2 = x + w
                y2 = y + h
                
                
                face_img = frame[y:y2, x:x2]
                
                if face_img.size == 0:
                    continue
                
                try:
                   
                    analysis = DeepFace.analyze(
                        img_path=face_img,
                        actions=['emotion'],
                        detector_backend='skip', 
                        enforce_detection=False
                    )
                    
                    # Get dominant emotion and emotion scores
                    emotion_dict = analysis[0]['emotion']
                    dominant_emotion = analysis[0]['dominant_emotion']
                    confidence = emotion_dict[dominant_emotion]
                    
                except Exception as e:
                    print("DeepFace error:", e)
                    dominant_emotion = "unknown"
                    confidence = 0
                
                # Store emotion for API response
                with lock:
                    current_emotions[idx] = {
                        "emotion": dominant_emotion,
                        "confidence": round(confidence, 2),
                        "coords": [x, y, x2, y2]
                    }
                
                # Draw on frame with better styling
                color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                
                # Add emotion label with confidence
                label = f"{dominant_emotion} ({confidence:.1f}%)"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame in the format required for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()


@app.route('/')
def index():
    """Render the main landing page"""
    return render_template('index.html')


@app.route('/face')
def face():
    """Render the live face detection page"""
    return render_template('face.html')


@app.route('/test')
def test():
    """Render the emotion quiz page"""
    return render_template('test.html')


@app.route('/practice')
def practice():
    """Render the practice mode page"""
    return render_template('practice.html')


@app.route('/progress')
def progress():
    """Render the progress tracking page"""
    return render_template('progress.html')


@app.route('/history')
def history():
    """Render the detection history page"""
    return render_template('history.html')


@app.route('/audio')
def audio_test():
    """Render the audio emotion detection test page"""
    return render_template('audio_test.html')


@app.route('/start_camera')
def start_camera():
    """Start the camera feed"""
    global camera_active
    camera_active = True
    return jsonify({"status": "Camera started"})


@app.route('/stop_camera')
def stop_camera():
    """Stop the camera feed"""
    global camera_active
    camera_active = False
    return jsonify({"status": "Camera stopped"})


@app.route('/video_feed')
def video_feed():
    """Stream video feed with emotion detection"""
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotions')
def get_emotions():
    """Get current detected emotions"""
    with lock:
        emotions = current_emotions.copy()
    return jsonify(emotions)


# ============================================
# AUDIO EMOTION DETECTION ROUTES
# ============================================

@app.route('/audio/record', methods=['POST'])
def record_audio():
    """Record audio from microphone and detect emotion"""
    try:
        duration = request.json.get('duration', 7)
        sample_rate = request.json.get('sample_rate', 22050)
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio_data = audio_data.flatten()
        
        # Get detector and make prediction
        detector = get_detector()
        if detector is None:
            return jsonify({
                'error': 'Audio detector not initialized'
            }), 500
        
        result = detector.predict_from_audio(audio_data, sample_rate)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'emotion': 'neutral',
            'confidence': 0.0
        }), 500


@app.route('/audio/predict_file', methods=['POST'])
def predict_audio_file():
    """Predict emotion from uploaded audio file"""
    try:
        logger.info("Received audio file upload request")
        
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        logger.info(f"Processing file: {file.filename}")
        
        # Save temporary file to system temp directory
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            temp_path = tmp.name
            file.save(temp_path)
            logger.info(f"Saved to temp: {temp_path}, size: {os.path.getsize(temp_path)} bytes")
        
        # Also save a copy for debugging in data folder
        import shutil
        debug_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(debug_dir, exist_ok=True)
        debug_files = len([f for f in os.listdir(debug_dir) if f.startswith('audio_')])
        debug_path = os.path.join(debug_dir, f'audio_{debug_files}.wav')
        try:
            shutil.copy(temp_path, debug_path)
            logger.info(f"Saved debug copy to: {debug_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug copy: {e}")
        
        # Get detector and make prediction
        detector = get_detector()
        if detector is None:
            logger.error("Audio detector not initialized")
            return jsonify({
                'error': 'Audio detector not initialized'
            }), 500
        
        logger.info("Making prediction...")
        result = detector.predict_from_file(temp_path)
        logger.info(f"Prediction result: {result}")
        
        # Clean up
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to clean temp file: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in predict_audio_file: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'emotion': 'neutral',
            'confidence': 0.0
        }), 500


@app.route('/audio/predict_numpy', methods=['POST'])
def predict_audio_numpy():
    """Predict emotion from numpy array of audio samples"""
    try:
        data = request.json
        audio_array = np.array(data.get('audio', []))
        sample_rate = data.get('sample_rate', 22050)
        
        if len(audio_array) == 0:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Get detector and make prediction
        detector = get_detector()
        if detector is None:
            return jsonify({
                'error': 'Audio detector not initialized'
            }), 500
        
        result = detector.predict_from_audio(audio_array, sample_rate)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'emotion': 'neutral',
            'confidence': 0.0
        }), 500


@app.route('/audio/info', methods=['GET'])
def audio_info():
    """Get information about audio emotion detection capabilities"""
    detector = get_detector()
    if detector is None:
        return jsonify({'error': 'Audio detector not initialized'}), 500
    
    return jsonify({
        'status': 'ready',
        'emotions': list(detector.model.classes_),
        'sample_rate': 22050,
        'feature_count': 26,  # 13 MFCC + 12 Chroma + 1 ZCR
        'duration_recommended': 7  # seconds
    })


if __name__ == '__main__':
    app.run(debug=False, threaded=True)
