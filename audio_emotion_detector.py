"""
Audio Emotion Detection Module
Loads trained SVM model and makes predictions on audio input
"""

import os
import joblib
import numpy as np
import logging
from audio_feature_extraction import extract_features_from_audio, extract_features

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where models are stored
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

class AudioEmotionDetector:
    """Audio emotion detection using pre-trained SVM model"""
    
    def __init__(self, model_path=None, scaler_path=None, label_encoder_path=None):
        """Initialize the audio emotion detector with pre-trained models"""
        
        # Use default paths if not provided
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "svm_model.pkl")
        if scaler_path is None:
            scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        if label_encoder_path is None:
            label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(label_encoder_path)
            logger.info("✅ Audio models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"❌ Model file not found: {e}")
            raise
    
    def predict_from_audio(self, audio_data, sample_rate=22050):
        """
        Predict emotion from raw audio data
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: sample rate of audio (default: 22050 Hz)
        
        Returns:
            dict: {
                'emotion': predicted emotion label,
                'confidence': confidence score (0-1),
                'probabilities': dict of all emotions with their probabilities
            }
        """
        try:
            logger.info(f"Starting prediction: audio_data shape={audio_data.shape}, sr={sample_rate}")
            logger.info(f"Audio statistics: min={np.min(audio_data):.6f}, max={np.max(audio_data):.6f}, mean={np.mean(audio_data):.6f}, std={np.std(audio_data):.6f}")
            
            # Extract features from audio
            features = extract_features_from_audio(audio_data, sample_rate)
            logger.info(f"Features extracted: shape={features.shape}")
            logger.info(f"Feature values: {features}")
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            logger.info(f"Features scaled: shape={features_scaled.shape}")
            logger.info(f"Scaled feature values: {features_scaled[0]}")
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            logger.info(f"Prediction: {prediction}, probabilities: {probabilities}")
            
            # Get confidence
            confidence = float(np.max(probabilities))
            logger.info(f"Confidence: {confidence}")
            
            # If confidence is extremely low, classify as neutral (but much lower threshold)
            if confidence < 0.15:
                logger.info(f"Confidence {confidence} < 0.15, setting to neutral")
                prediction = "neutral"
                confidence = 0.15
            
            # Create probability dictionary with clean string keys
            prob_dict = {}
            for label, prob in zip(self.model.classes_, probabilities):
                # Convert numpy string to regular string
                label_str = str(label).strip()
                prob_dict[label_str] = float(prob)
            
            # Convert prediction to regular string
            prediction_str = str(prediction).strip()
            
            result = {
                'emotion': prediction_str,
                'confidence': round(confidence, 4),
                'probabilities': prob_dict
            }
            logger.info(f"Final result: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'probabilities': {},
                'error': str(e)
            }
    
    def predict_from_file(self, file_path):
        """
        Predict emotion from audio file
        
        Args:
            file_path: path to audio file
        
        Returns:
            dict: prediction results
        """
        try:
            import librosa
            logger.info(f"Loading audio file: {file_path}")
            
            # Try to load with librosa - handles multiple formats
            try:
                audio_data, sr = librosa.load(file_path, sr=22050, mono=True)
                logger.info(f"Audio loaded successfully: shape={audio_data.shape}, sr={sr}")
            except Exception as load_error:
                logger.error(f"Librosa load error: {load_error}", exc_info=True)
                # Try alternative: scipy
                try:
                    from scipy.io import wavfile
                    sr, audio_data = wavfile.read(file_path)
                    logger.info(f"Loaded with scipy: sr={sr}, shape={audio_data.shape}")
                    # Convert to float and resample if needed
                    audio_data = audio_data.astype(float) / 32768.0
                    if sr != 22050:
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=22050)
                        sr = 22050
                except Exception as scipy_error:
                    logger.error(f"Scipy load also failed: {scipy_error}", exc_info=True)
                    raise load_error
            
            result = self.predict_from_audio(audio_data, sr)
            logger.info(f"Prediction complete: {result}")
            return result
        except Exception as e:
            logger.error(f"Error loading audio file: {e}", exc_info=True)
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }


# Initialize detector globally (lazy loading)
_detector = None

def get_detector():
    """Get or initialize the audio emotion detector"""
    global _detector
    if _detector is None:
        try:
            _detector = AudioEmotionDetector()
        except Exception as e:
            logger.error(f"Failed to initialize audio detector: {e}")
            return None
    return _detector
