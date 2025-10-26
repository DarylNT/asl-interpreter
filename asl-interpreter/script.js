const WS_URL = "ws://127.0.0.1:8000/ws";

// DOM Elements
const video = document.getElementById("video");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const predictedLetterEl = document.getElementById("predicted_letter");
const confidenceEl = document.getElementById("confidence");
const predictionModeEl = document.getElementById("prediction_mode");

// Global state
let stream = null;
let ws = null;
let hands = null;
let camera = null;
let landmarksBuffer = [];
const BUFFER_SIZE = 10;

function setStatus(text, cls = "") {
    statusEl.textContent = text;
    statusEl.className = "status" + (cls ? (" " + cls) : "");
}

function fitCanvas() {
    const r = video.getBoundingClientRect();
    canvas.width = r.width * devicePixelRatio;
    canvas.height = r.height * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}

function updatePrediction(letter, confidence, isTemporalMode = false) {
    predictedLetterEl.textContent = letter;
    confidenceEl.textContent = `Confidence: ${confidence.toFixed(1)}%`;
    predictionModeEl.textContent = isTemporalMode ? "Temporal Enhanced" : "Standard Mode";
    predictionModeEl.style.color = isTemporalMode ? 'var(--ok)' : 'var(--muted)';
}

function normalizeWorldLandmarks(landmarks) {
    const points = landmarks.map(lm => [lm.x, lm.y, lm.z]);
    const mean = points.reduce((acc, p) => acc.map((v, i) => v + p[i]), [0, 0, 0])
                      .map(v => v / points.length);
    const centered = points.map(p => p.map((v, i) => v - mean[i]));
    const std = Math.sqrt(centered.reduce((acc, p) => acc + p.reduce((sum, v) => sum + v * v, 0), 0) / (points.length * 3));
    return centered.map(p => p.map(v => v / (std + 1e-8)));
}

async function startCamera() {
    try {
        hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        camera = new Camera(video, {
            onFrame: async () => {
                await hands.send({image: video});
            },
            width: 960,
            height: 540
        });

        hands.onResults(onResults);
        await camera.start();
        
        connectWS();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        setStatus("Camera running", "ok");
        
        fitCanvas();
        new ResizeObserver(fitCanvas).observe(document.body);
    } catch(e) {
        setStatus("Camera error: " + e.message, "err");
    }
}

function stopCamera() {
    if (camera) {
        camera.stop();
    }
    if (hands) {
        hands.close();
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close(1000, "bye");
    }
    
    ws = null;
    hands = null;
    camera = null;
    landmarksBuffer = [];
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    updatePrediction("-", 0);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus("Idle");
}

function connectWS() {
    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    
    ws.onopen = () => {
        setStatus("Connected to prediction server", "ok");
    };
    
    ws.onclose = () => {
        setStatus("Connection closed");
    };
    
    ws.onerror = () => {
        setStatus("WebSocket error", "err");
    };
    
    ws.onmessage = (e) => {
        try {
                        const data = JSON.parse(e.data);
            if (data.letter && data.confidence) {
                updatePrediction(
                    data.letter,
                    data.confidence,
                    data.temporal_mode || false
                );
            }
        } catch(err) {
            console.error("Error parsing prediction:", err);
        }
    };
}

function onResults(results) {
    // Clear canvas
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw camera feed
    if (results.multiHandLandmarks) {
        // Draw hand landmarks
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                color: '#00FF00',
                lineWidth: 2
            });
            drawLandmarks(ctx, landmarks, {
                color: '#FF0000',
                lineWidth: 1,
                radius: 3
            });
        }

        // Send world landmarks for prediction
        if (results.multiHandWorldLandmarks && results.multiHandWorldLandmarks.length > 0 && ws?.readyState === WebSocket.OPEN) {
            const worldLandmarks = results.multiHandWorldLandmarks[0].landmarks;
            const normalizedLandmarks = normalizeWorldLandmarks(worldLandmarks);
            
            ws.send(JSON.stringify({
                landmarks: normalizedLandmarks
            }));
        }
    }

    ctx.restore();
}

// Event Listeners
startBtn.onclick = startCamera;
stopBtn.onclick = stopCamera;

// Handle page unload
window.onbeforeunload = () => {
    stopCamera();
};
            drawOverlay(d);
            labelEl.textContent = d.prediction || "—";
        }catch{}
    };
}

function pumpFrames(){
    const tmp = document.createElement("canvas"), tctx = tmp.getContext("2d");
    let last=0; const target=100; // ~10 fps
    const loop = (ts)=>{
        sendLoop=requestAnimationFrame(loop);
        if(!ws || ws.readyState!==WebSocket.OPEN) return;
        if(ts-last<target) return; last=ts;
        const w = 320, h = Math.round((video.videoHeight/video.videoWidth)*w)||180;
        tmp.width=w; tmp.height=h; tctx.drawImage(video,0,0,w,h);
        tmp.toBlob(b=>{ if(b && ws && ws.readyState===WebSocket.OPEN) ws.send(b); },"image/jpeg",0.7);
    };
    sendLoop=requestAnimationFrame(loop);
}
function drawOverlay(d){
    // Clear the canvas and show nothing
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}


startBtn.addEventListener("click", startCamera);
stopBtn .addEventListener("click", stopCamera);
