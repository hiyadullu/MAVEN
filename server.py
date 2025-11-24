import uvicorn
import tempfile
import soundfile as sf
import asyncio
from fastapi import FastAPI, WebSocket
from speechbrain.inference.interfaces import foreign_class

app = FastAPI()

# Load pretrained model
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="classification.py",
    classname="SBEmotionRecognition"
)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    while True:
        audio_bytes = await ws.receive_bytes()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            filename = f.name

        try:
            emotion = classifier.classify_file(filename)[3]
            await ws.send_text(emotion)
        except:
            await ws.send_text("error")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
