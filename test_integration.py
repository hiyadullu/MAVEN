"""
Test script for audio emotion detection integration
"""

import json
import requests
import time

BASE_URL = "http://localhost:5000"

def test_audio_info():
    """Test getting audio detector capabilities"""
    print("\n🔍 Testing /audio/info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/audio/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_audio_predict_numpy():
    """Test predicting emotion from numpy array"""
    print("\n🔍 Testing /audio/predict_numpy endpoint...")
    try:
        # Create a simple test audio array
        import numpy as np
        duration = 7
        sample_rate = 22050
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).tolist()
        
        payload = {
            "audio": audio,
            "sample_rate": sample_rate
        }
        
        response = requests.post(f"{BASE_URL}/audio/predict_numpy", 
                               json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_audio_file():
    """Test predicting emotion from file (if you have a test audio file)"""
    print("\n🔍 Testing /audio/predict_file endpoint...")
    try:
        # Try to use an audio file if available
        import os
        test_file = "test_audio.wav"
        
        if not os.path.exists(test_file):
            print("⚠️  No test audio file found. Skipping...")
            return None
        
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/audio/predict_file",
                                   files=files)
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_facial_endpoints():
    """Test facial emotion detection endpoints"""
    print("\n🔍 Testing facial emotion detection endpoints...")
    try:
        # Test start camera
        print("  - Testing /start_camera...")
        response = requests.get(f"{BASE_URL}/start_camera")
        print(f"    Status: {response.status_code}, Response: {response.json()}")
        
        time.sleep(1)
        
        # Test get emotions
        print("  - Testing /emotions...")
        response = requests.get(f"{BASE_URL}/emotions")
        print(f"    Status: {response.status_code}, Response: {response.json()}")
        
        # Test stop camera
        print("  - Testing /stop_camera...")
        response = requests.get(f"{BASE_URL}/stop_camera")
        print(f"    Status: {response.status_code}, Response: {response.json()}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_page_routes():
    """Test that all page routes are accessible"""
    print("\n🔍 Testing page routes...")
    routes = ['/', '/face', '/test', '/practice', '/progress', '/history']
    
    for route in routes:
        try:
            response = requests.get(f"{BASE_URL}{route}")
            status = "✅" if response.status_code == 200 else "❌"
            print(f"  {status} {route}: {response.status_code}")
        except Exception as e:
            print(f"  ❌ {route}: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("EmotiLearn Audio + Video Integration Tests")
    print("=" * 60)
    
    try:
        # Check if server is running
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server is running on {BASE_URL}")
    except:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print("   Please start the Flask app with: python app.py")
        return
    
    # Run tests
    results = {
        "Audio Info": test_audio_info(),
        "Audio Prediction (NumPy)": test_audio_predict_numpy(),
        "Audio File": test_audio_file(),
        "Facial Detection": test_facial_endpoints(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{status}: {test_name}")
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    print(f"\nOverall: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
