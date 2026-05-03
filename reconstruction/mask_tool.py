"""
mask_tool.py
────────────
Browser-based interactive mask painter.
No display / X11 / Qt / GTK required — works over SSH.

Usage
─────
  python mask_tool.py --ref ref.jpg --broken broken.jpg --output_dir ./outputs

Then open  http://localhost:5000  in your browser.

Paint the SOURCE mask on the ref image, click "Save Source Mask".
Paint the TARGET mask on the broken image, click "Save Target Mask".
Both masks are saved as PNG files in --output_dir.

Pass the saved masks to reconstruct_splats.py:
  python reconstruct_splats.py ... --src_mask outputs/src_mask.png \
                                   --tgt_mask outputs/tgt_mask.png
"""

import os
import sys
import argparse
import base64
import io
import numpy as np
import cv2
from flask import Flask, render_template_string, request, jsonify
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--ref",        type=str, required=True)
parser.add_argument("--broken",     type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./outputs")
parser.add_argument("--max_dim",    type=int, default=800)
parser.add_argument("--port",       type=int, default=5000)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_and_encode(path, max_dim):
    bgr   = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w  = bgr.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64    = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}", bgr.shape[1], bgr.shape[0]

ref_data,    ref_w,    ref_h    = load_and_encode(args.ref,    args.max_dim)
broken_data, broken_w, broken_h = load_and_encode(args.broken, args.max_dim)

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Mask Painter</title>
<style>
  body { margin:0; background:#1a1a2e; color:#eee; font-family:sans-serif; display:flex; flex-direction:column; align-items:center; }
  h2   { margin:12px 0 4px; font-size:1.1em; color:#0ff; }
  .controls { display:flex; gap:12px; align-items:center; margin:8px 0; flex-wrap:wrap; justify-content:center; }
  button { padding:7px 16px; border:none; border-radius:5px; cursor:pointer; font-size:0.9em; font-weight:bold; }
  .btn-clear  { background:#c0392b; color:#fff; }
  .btn-save   { background:#27ae60; color:#fff; }
  .btn-undo   { background:#2980b9; color:#fff; }
  label  { font-size:0.88em; }
  input[type=range] { width:100px; }
  .canvas-wrap { position:relative; display:inline-block; border:2px solid #444; }
  canvas { display:block; cursor:crosshair; }
  .status { margin:6px 0 2px; font-size:0.85em; color:#0f0; min-height:1.2em; }
  .section { margin-bottom:18px; }
  select { background:#333; color:#eee; border:1px solid #555; padding:4px 8px; border-radius:4px; }
</style>
</head>
<body>
<h1 style="margin:14px 0 4px;font-size:1.3em;">Gaussian Splat Mask Painter</h1>
<p style="margin:0 0 8px;font-size:0.82em;color:#aaa;">Paint masks, then click Save. Brush draws; Eraser removes. Undo removes last stroke.</p>

<div class="section">
  <h2>SOURCE — ref image &nbsp;<span style="color:#aaa;font-size:0.8em;">(paint the region to copy from)</span></h2>
  <div class="controls">
    <label>Tool:
      <select id="tool_src">
        <option value="brush">Brush</option>
        <option value="eraser">Eraser</option>
      </select>
    </label>
    <label>Size: <input type="range" id="size_src" min="2" max="80" value="20"> <span id="size_src_val">20</span>px</label>
    <label>Opacity: <input type="range" id="opacity_src" min="10" max="100" value="50"> <span id="opacity_src_val">50</span>%</label>
    <button class="btn-undo"  onclick="undo('src')">↩ Undo</button>
    <button class="btn-clear" onclick="clearMask('src')">✕ Clear</button>
    <button class="btn-save"  onclick="saveMask('src')">💾 Save Source Mask</button>
  </div>
  <div class="canvas-wrap">
    <canvas id="img_src"  width="{{ ref_w }}" height="{{ ref_h }}"></canvas>
    <canvas id="mask_src" width="{{ ref_w }}" height="{{ ref_h }}"
            style="position:absolute;top:0;left:0;opacity:0.55;"></canvas>
  </div>
  <div class="status" id="status_src">Not saved yet.</div>
</div>

<div class="section">
  <h2>TARGET — broken image &nbsp;<span style="color:#aaa;font-size:0.8em;">(paint the damaged region to fill)</span></h2>
  <div class="controls">
    <label>Tool:
      <select id="tool_brk">
        <option value="brush">Brush</option>
        <option value="eraser">Eraser</option>
      </select>
    </label>
    <label>Size: <input type="range" id="size_brk" min="2" max="80" value="20"> <span id="size_brk_val">20</span>px</label>
    <label>Opacity: <input type="range" id="opacity_brk" min="10" max="100" value="50"> <span id="opacity_brk_val">50</span>%</label>
    <button class="btn-undo"  onclick="undo('brk')">↩ Undo</button>
    <button class="btn-clear" onclick="clearMask('brk')">✕ Clear</button>
    <button class="btn-save"  onclick="saveMask('brk')">💾 Save Target Mask</button>
  </div>
  <div class="canvas-wrap">
    <canvas id="img_brk"  width="{{ broken_w }}" height="{{ broken_h }}"></canvas>
    <canvas id="mask_brk" width="{{ broken_w }}" height="{{ broken_h }}"
            style="position:absolute;top:0;left:0;opacity:0.55;"></canvas>
  </div>
  <div class="status" id="status_brk">Not saved yet.</div>
</div>

<script>
const IMAGES = {
  src: "{{ ref_data }}",
  brk: "{{ broken_data }}"
};

// ── state per canvas ──────────────────────────────────────────────────────────
const state = {
  src: { drawing: false, history: [] },
  brk: { drawing: false, history: [] }
};

// ── initialise canvases ───────────────────────────────────────────────────────
function init(id) {
  const imgCanvas  = document.getElementById(`img_${id}`);
  const maskCanvas = document.getElementById(`mask_${id}`);
  const imgCtx     = imgCanvas.getContext("2d");
  const maskCtx    = maskCanvas.getContext("2d");

  // draw background image
  const img = new Image();
  img.onload = () => imgCtx.drawImage(img, 0, 0);
  img.src    = IMAGES[id];

  // clear mask with transparent
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

  // mouse events on the mask canvas
  maskCanvas.addEventListener("mousedown",  e => startStroke(e, id, maskCanvas, maskCtx));
  maskCanvas.addEventListener("mousemove",  e => continueStroke(e, id, maskCanvas, maskCtx));
  maskCanvas.addEventListener("mouseup",    e => endStroke(id, maskCtx));
  maskCanvas.addEventListener("mouseleave", e => endStroke(id, maskCtx));

  // touch support
  maskCanvas.addEventListener("touchstart",  e => { e.preventDefault(); startStroke(e.touches[0], id, maskCanvas, maskCtx); }, {passive:false});
  maskCanvas.addEventListener("touchmove",   e => { e.preventDefault(); continueStroke(e.touches[0], id, maskCanvas, maskCtx); }, {passive:false});
  maskCanvas.addEventListener("touchend",    e => endStroke(id, maskCtx));
}

function getPos(e, canvas) {
  const r = canvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) * canvas.width  / r.width,
           y: (e.clientY - r.top)  * canvas.height / r.height };
}

function startStroke(e, id, canvas, ctx) {
  state[id].drawing = true;
  // snapshot before stroke for undo
  state[id].history.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  if (state[id].history.length > 30) state[id].history.shift();
  const pos  = getPos(e, canvas);
  paintAt(ctx, pos.x, pos.y, id);
}

function continueStroke(e, id, canvas, ctx) {
  if (!state[id].drawing) return;
  const pos = getPos(e, canvas);
  paintAt(ctx, pos.x, pos.y, id);
}

function endStroke(id, ctx) {
  state[id].drawing = false;
}

function paintAt(ctx, x, y, id) {
  const tool    = document.getElementById(`tool_${id}`).value;
  const size    = parseInt(document.getElementById(`size_${id}`).value);
  const opacity = parseInt(document.getElementById(`opacity_${id}`).value) / 100;

  if (tool === "eraser") {
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0,0,0,1)";
    ctx.fill();
    ctx.restore();
  } else {
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(0,200,255,${opacity})`;
    ctx.fill();
    ctx.restore();
  }
}

function clearMask(id) {
  const c = document.getElementById(`mask_${id}`);
  const ctx = c.getContext("2d");
  state[id].history.push(ctx.getImageData(0,0,c.width,c.height));
  ctx.clearRect(0, 0, c.width, c.height);
}

function undo(id) {
  if (!state[id].history.length) return;
  const c = document.getElementById(`mask_${id}`);
  const ctx = c.getContext("2d");
  ctx.putImageData(state[id].history.pop(), 0, 0);
}

async function saveMask(id) {
  const label    = id === "src" ? "source" : "target";
  const statusEl = document.getElementById(`status_${id}`);
  statusEl.style.color = "#ff0";
  statusEl.textContent = "Saving…";

  // render mask canvas to grayscale PNG:
  // wherever the user painted (any alpha>0) → white, else black
  const maskCanvas = document.getElementById(`mask_${id}`);
  const w = maskCanvas.width, h = maskCanvas.height;
  const srcData = maskCanvas.getContext("2d").getImageData(0,0,w,h);

  const offscreen = document.createElement("canvas");
  offscreen.width = w; offscreen.height = h;
  const octx = offscreen.getContext("2d");
  const dst   = octx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const alpha = srcData.data[i*4+3];
    const val   = alpha > 10 ? 255 : 0;
    dst.data[i*4]   = val;
    dst.data[i*4+1] = val;
    dst.data[i*4+2] = val;
    dst.data[i*4+3] = 255;
  }
  octx.putImageData(dst, 0, 0);

  const pngData = offscreen.toDataURL("image/png");

  try {
    const resp = await fetch("/save_mask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ id, data: pngData })
    });
    const json = await resp.json();
    if (json.ok) {
      statusEl.style.color = "#0f0";
      statusEl.textContent = `✅ Saved → ${json.path}`;
    } else {
      statusEl.style.color = "#f00";
      statusEl.textContent = `❌ Error: ${json.error}`;
    }
  } catch(err) {
    statusEl.style.color = "#f00";
    statusEl.textContent = `❌ Fetch error: ${err}`;
  }
}

// wire up range display labels
["src","brk"].forEach(id => {
  ["size","opacity"].forEach(ctrl => {
    const el = document.getElementById(`${ctrl}_${id}`);
    const lbl = document.getElementById(`${ctrl}_${id}_val`);
    el.addEventListener("input", () => lbl.textContent = el.value);
  });
  init(id);
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(
        HTML,
        ref_data=ref_data,     ref_w=ref_w,     ref_h=ref_h,
        broken_data=broken_data, broken_w=broken_w, broken_h=broken_h,
    )

@app.route("/save_mask", methods=["POST"])
def save_mask():
    try:
        payload   = request.get_json()
        mask_id   = payload["id"]           # "src" or "brk"
        data_url  = payload["data"]         # "data:image/png;base64,..."

        header, b64 = data_url.split(",", 1)
        img_bytes   = base64.b64decode(b64)
        pil_img     = Image.open(io.BytesIO(img_bytes)).convert("L")
        arr         = np.array(pil_img)

        # threshold: any non-black pixel → 255
        arr = (arr > 10).astype(np.uint8) * 255

        fname = "src_mask.png" if mask_id == "src" else "tgt_mask.png"
        fpath = os.path.join(args.output_dir, fname)
        cv2.imwrite(fpath, arr)

        print(f"  Saved {fname}  ({arr.sum()//255} px masked)  → {fpath}")
        return jsonify({"ok": True, "path": fpath})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)})

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Mask Painter ready.")
    print(f"  Open in your browser:  http://localhost:{args.port}")
    print(f"  Masks will be saved to: {os.path.abspath(args.output_dir)}/")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)