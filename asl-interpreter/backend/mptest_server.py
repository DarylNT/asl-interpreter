# mptest_server.py — WebSocket server: receives JPEGs, returns JSON
# pip install: fastapi uvicorn pillow numpy opencv-python torch mediapipe
import io, json
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from mptest_lib import process_frame

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"]
                   )

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            blob = await ws.receive_bytes()
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            arr = np.array(img)
            pred, lms, w, h = process_frame(arr)
            await ws.send_text(json.dumps({
                "prediction": pred, "landmarks": lms, "width": w, "height": h
            }))
    finally:
        try: await ws.close()
        except: pass

if __name__ == "__main__":
    uvicorn.run("mptest_server:app", host="127.0.0.1", port=8000, reload=False)
