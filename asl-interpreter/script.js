const WS_URL = "ws://127.0.0.1:8000/ws";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn  = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const labelEl  = document.getElementById("label");

let stream=null, ws=null, sendLoop=null;

function setStatus(t, cls=""){ statusEl.textContent=t; statusEl.className="status"+(cls?(" "+cls):""); }
function fitCanvas(){
    const r = video.getBoundingClientRect();
    overlay.width = r.width*devicePixelRatio;
    overlay.height= r.height*devicePixelRatio;
    ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}

async function startCamera(){
    try{
        stream = await navigator.mediaDevices.getUserMedia({
            video:{facingMode:"user", width:{ideal:960}, height:{ideal:540}}, audio:false
        });
        video.srcObject = stream; await video.play();
        fitCanvas(); new ResizeObserver(fitCanvas).observe(document.body);
        connectWS();
        startBtn.disabled = true; stopBtn.disabled = false;
        setStatus("Camera on. Connecting to local Python…");
    }catch(e){ setStatus("Camera error: "+e.message, "err"); }
}

function stopCamera(){
    if(sendLoop){ cancelAnimationFrame(sendLoop); sendLoop=null; }
    if(ws && ws.readyState===WebSocket.OPEN) ws.close(1000,"bye");
    ws=null;
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
    video.srcObject=null; ctx.clearRect(0,0,overlay.width,overlay.height);
    labelEl.textContent="—"; startBtn.disabled=false; stopBtn.disabled=true; setStatus("Idle");
}

function connectWS(){
    ws = new WebSocket(WS_URL); ws.binaryType="arraybuffer";
    ws.onopen = ()=>{ setStatus("Connected to local Python.","ok"); pumpFrames(); };
    ws.onclose= ()=> setStatus("Connection closed.");
    ws.onerror= ()=> setStatus("WebSocket error.","err");
    ws.onmessage = (e)=>{
        try{
            const d = JSON.parse(e.data);
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
