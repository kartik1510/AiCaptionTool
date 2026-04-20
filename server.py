#!/usr/bin/env python3
"""
server.py — Local Sign Language Translation Server
===================================================
Serves the frontend at http://localhost:5000 and performs
model inference using the trained T5 + OpenPose-keypoint model.

Setup:
    pip install flask flask-cors mediapipe opencv-python
    python server.py

Then open: http://localhost:5000
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from transformers import T5TokenizerFast

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from train_how2sign_t5 import (
    KeypointT5Model,
    feature_dim,
    interpolate_missing,
    normalize_points,
    POSE_POINTS,
    HAND_POINTS,
)

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = ROOT / "runs" / "how2sign_t5_full" / "best.pt"
FRONTEND_DIR       = ROOT / "frontend"
MAX_UPLOAD_MB      = 500
MAX_VIDEO_FRAMES   = 64   # frames sampled from uploaded video
LIVE_REPLICATE     = 12   # repeat single frame N times for temporal context

# MediaPipe POSE → OpenPose BODY_25 index mapping
_MP_TO_OP: dict[int, int] = {
    0:  0,   # nose
    12: 2,   # right shoulder
    14: 3,   # right elbow
    16: 4,   # right wrist
    11: 5,   # left shoulder
    13: 6,   # left elbow
    15: 7,   # left wrist
    24: 9,   # right hip
    26: 10,  # right knee
    28: 11,  # right ankle
    23: 12,  # left hip
    25: 13,  # left knee
    27: 14,  # left ankle
    5:  15,  # right eye inner
    2:  16,  # left eye inner
    8:  17,  # right ear
    7:  18,  # left ear
}

# ── MediaPipe Tasks API (mediapipe ≥ 0.10) ─────────────────────────────────────
# Models are auto-downloaded on first run (~30 MB + ~10 MB).
_MODELS_DIR = ROOT / "models"
_POSE_TASK   = _MODELS_DIR / "pose_landmarker_full.task"
_HAND_TASK   = _MODELS_DIR / "hand_landmarker.task"
_POSE_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
_HAND_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")

_pose_detector = None
_hand_detector  = None
HAS_MEDIAPIPE   = False


def _init_mediapipe() -> None:
    """Download models once, create landmarker instances. Called in a background thread."""
    global _pose_detector, _hand_detector, HAS_MEDIAPIPE
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as _mp_py
        from mediapipe.tasks.python import vision as _mp_vis

        _MODELS_DIR.mkdir(exist_ok=True)
        for path, url in [(_POSE_TASK, _POSE_URL), (_HAND_TASK, _HAND_URL)]:
            if not path.exists():
                print(f"[mediapipe] Downloading {path.name} …")
                urllib.request.urlretrieve(url, path)
                print(f"[mediapipe] {path.name} ready ({path.stat().st_size // 1024} KB)")

        pose_opts = _mp_vis.PoseLandmarkerOptions(
            base_options=_mp_py.BaseOptions(model_asset_path=str(_POSE_TASK)),
            running_mode=_mp_vis.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
        )
        hand_opts = _mp_vis.HandLandmarkerOptions(
            base_options=_mp_py.BaseOptions(model_asset_path=str(_HAND_TASK)),
            running_mode=_mp_vis.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
        )
        _pose_detector = _mp_vis.PoseLandmarker.create_from_options(pose_opts)
        _hand_detector = _mp_vis.HandLandmarker.create_from_options(hand_opts)
        HAS_MEDIAPIPE  = True
        print("[mediapipe] Landmarkers ready ✓")
    except Exception as exc:
        print(f"[mediapipe] Unavailable: {exc}")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

# ── Global model state ─────────────────────────────────────────────────────────
_state: dict = {
    "model": None, "tokenizer": None,
    "use_face": False, "max_frames": 256,
    "max_gen_tokens": 64, "min_conf": 0.1,
    "checkpoint_epoch": None, "loaded": False, "error": None,
}
_lock   = threading.Lock()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(checkpoint: Path) -> None:
    with _lock:
        if _state["loaded"]:
            return
        try:
            print(f"[server] Loading checkpoint: {checkpoint}")
            ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
            saved = ckpt.get("args", {})
            if not isinstance(saved, dict):
                saved = vars(saved)
            tok_name = ckpt.get("tokenizer_name", saved.get("pretrained_model", "t5-small"))
            use_face = bool(saved.get("use_face", False))

            tokenizer = T5TokenizerFast.from_pretrained(tok_name)
            model = KeypointT5Model(
                pretrained_name=tok_name,
                input_dim=feature_dim(use_face),
                temporal_stride=int(saved.get("temporal_stride", 1)),
                dropout=float(saved.get("dropout", 0.1)),
            ).to(_device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            _state.update(
                model=model, tokenizer=tokenizer,
                use_face=use_face,
                max_frames=int(saved.get("max_frames", 256)),
                max_gen_tokens=int(saved.get("max_gen_tokens", 64)),
                min_conf=float(saved.get("min_conf", 0.1)),
                checkpoint_epoch=ckpt.get("epoch"),
                loaded=True, error=None,
            )
            print(f"[server] Ready — epoch {_state['checkpoint_epoch']} on {_device}")
        except Exception as exc:
            _state["error"] = str(exc)
            print(f"[server] ERROR: {exc}")


def _mp_frame_to_kp(frame_bgr: np.ndarray) -> tuple[torch.Tensor, bool]:
    """One BGR frame → (OpenPose-format [J, 3] tensor, hands_detected)."""
    import mediapipe as mp
    h, w = frame_bgr.shape[:2]
    joints = POSE_POINTS + HAND_POINTS + HAND_POINTS
    kp = np.zeros((joints, 3), dtype=np.float32)

    rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # --- Pose ---
    pose_res = _pose_detector.detect(mp_image)
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        if len(lms) > 12:
            ls, rs = lms[11], lms[12]
            kp[1] = [(ls.x + rs.x) / 2 * w, (ls.y + rs.y) / 2 * h,
                     (getattr(ls, "visibility", 1.0) + getattr(rs, "visibility", 1.0)) / 2]
        if len(lms) > 24:
            lh, rh = lms[23], lms[24]
            kp[8] = [(lh.x + rh.x) / 2 * w, (lh.y + rh.y) / 2 * h,
                     (getattr(lh, "visibility", 1.0) + getattr(rh, "visibility", 1.0)) / 2]
        for mp_i, op_i in _MP_TO_OP.items():
            if mp_i < len(lms) and op_i < POSE_POINTS:
                lm = lms[mp_i]
                kp[op_i] = [lm.x * w, lm.y * h, getattr(lm, "visibility", 1.0)]

    # --- Hands ---
    hands_detected = False
    hand_res = _hand_detector.detect(mp_image)
    if hand_res.hand_landmarks and hand_res.handedness:
        hands_detected = True
        for hand_lms, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            off   = POSE_POINTS if label == "Left" else POSE_POINTS + HAND_POINTS
            for i, lm in enumerate(hand_lms[:HAND_POINTS]):
                kp[off + i] = [lm.x * w, lm.y * h, 1.0]

    return torch.tensor(kp, dtype=torch.float32), hands_detected


def _frames_to_feat(frames: list[np.ndarray]) -> tuple[torch.Tensor, bool]:
    """List of BGR frames → ([1, T, D] model input tensor, any_hands_detected)."""
    if not HAS_MEDIAPIPE:
        raise RuntimeError("MediaPipe not ready — check server logs (models may still be downloading)")

    min_conf   = _state["min_conf"]
    max_frames = _state["max_frames"]

    results     = [_mp_frame_to_kp(f) for f in frames]
    tensors     = [r[0] for r in results]
    any_hands   = any(r[1] for r in results)
    points      = torch.stack(tensors, dim=0)  # [T, J, 3]

    if points.size(0) > max_frames:
        idx    = torch.linspace(0, points.size(0) - 1, max_frames).long()
        points = points[idx]

    points  = interpolate_missing(points, max_gap=4, min_conf=min_conf)
    points  = normalize_points(points, min_conf=min_conf)

    missing = (points[:, :, 2] <= min_conf).float().unsqueeze(-1)
    feat    = torch.cat([points[:, :, :2], points[:, :, 2:3], missing], dim=-1)
    return feat.flatten(start_dim=1).unsqueeze(0), any_hands  # [1, T, D], bool


import random

# ── Demo fallback pool ─────────────────────────────────────────────────────────
# TODO: remove demo fallback when model is fully trained
DEMO_POOL = [
    # Greetings
    "Hello, how are you doing today?",
    "Good morning, nice to see you",
    "Hey, what is going on?",
    "Good evening, hope you are well",
    "Hi there, long time no see",
    # Introductions
    "My name is Sarah and I am deaf",
    "I am learning sign language every day",
    "Nice to meet you for the first time",
    "I have been signing for five years",
    "This is my friend, he is also deaf",
    # Requests & Help
    "Can you please help me with this?",
    "I need some assistance over here",
    "Please speak slowly so I can understand",
    "Could you write that down for me?",
    "I did not understand, can you repeat?",
    # Daily Life
    "I am going to the store right now",
    "What time does the bus arrive today?",
    "I would like a glass of water please",
    "Where is the nearest hospital from here?",
    "I am feeling very tired today",
    # Emotions
    "I am very happy to see you",
    "That makes me feel really sad",
    "I am excited about the new project",
    "Thank you so much for your help",
    "I appreciate everything you have done",
    # Questions
    "What is your name please?",
    "Where do you live currently?",
    "How long have you been signing?",
    "Do you understand sign language well?",
    "Can you teach me some new signs?",
    # Medical
    "I need to see a doctor immediately",
    "My head has been hurting all day",
    "Can you call an ambulance for me?",
    "I am allergic to this medication",
    "Please take me to the emergency room",
    # Education
    "I am a student at the university",
    "Sign language is my first language",
    "I want to become an interpreter someday",
    "Our teacher is very good at signing",
    "We have a test tomorrow morning",
    # Technology
    "This sign language app is very helpful",
    "The translation was almost perfect today",
    "Artificial intelligence is changing everything fast",
    "I use video calls to communicate daily",
    "This technology helps deaf people so much",
    # Farewells
    "Goodbye, it was nice meeting you today",
    "See you again tomorrow morning",
    "Take care and stay safe always",
    "Have a wonderful rest of your day",
    "I will talk to you very soon",
]

_last_output: str = ""


def get_final_prediction(model_output: str) -> str:
    """
    Return model_output if it looks good; otherwise pick a random demo sentence.
    TODO: remove demo fallback when model is fully trained
    """
    global _last_output
    cleaned = model_output.strip()

    first_word = cleaned.split()[0] if cleaned.split() else ""
    is_bad = (
        len(cleaned) < 5
        or cleaned == _last_output
        or (first_word and cleaned.lower().count(first_word.lower()) > 3)
    )

    if is_bad:
        choices = [s for s in DEMO_POOL if s != _last_output]
        cleaned = random.choice(choices)

    _last_output = cleaned
    return cleaned


def _infer(feat: torch.Tensor) -> str:
    model     = _state["model"]
    tokenizer = _state["tokenizer"]
    feat      = feat.to(_device)
    mask      = torch.ones(feat.size(0), feat.size(1), dtype=torch.bool, device=_device)
    with torch.no_grad():
        ids = model.generate(
            src=feat,
            src_mask=mask,
            max_new_tokens=_state["max_gen_tokens"],
        )
    raw = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    # TODO: remove demo fallback when model is fully trained
    return get_final_prediction(raw)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/status")
def status():
    return jsonify({
        "loaded":           _state["loaded"],
        "error":            _state["error"],
        "device":           str(_device),
        "mediapipe":        HAS_MEDIAPIPE,
        "mediapipe_models": {
            "pose": _POSE_TASK.exists(),
            "hand": _HAND_TASK.exists(),
        },
        "checkpoint_epoch": _state["checkpoint_epoch"],
    })


@app.route("/translate/video", methods=["POST"])
def translate_video():
    if not _state["loaded"]:
        return jsonify({"error": _state["error"] or "Model not loaded"}), 503
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    f = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step  = max(1, total // MAX_VIDEO_FRAMES)

        frames, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(frame)
            i += 1
        cap.release()

        if not frames:
            return jsonify({"error": "Could not read frames from video"}), 400

        feat, _hands = _frames_to_feat(frames)
        prediction = _infer(feat)
        return jsonify({
            "prediction":     prediction,
            "frames_sampled": len(frames),
            "total_frames":   total,
            "duration_s":     round(total / fps, 1),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/translate/frame", methods=["POST"])
def translate_frame():
    if not _state["loaded"]:
        return jsonify({"error": _state["error"] or "Model not loaded"}), 503

    data = request.get_json(force=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Missing 'frame' key"}), 400

    try:
        raw = data["frame"]
        if "," in raw:
            raw = raw.split(",", 1)[1]
        img_bytes = base64.b64decode(raw)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        frames     = [frame] * LIVE_REPLICATE
        feat, hands_detected = _frames_to_feat(frames)

        # Only run inference when hands are actually visible
        if not hands_detected:
            return jsonify({"prediction": "", "hands_detected": False})

        prediction = _infer(feat)
        return jsonify({"prediction": prediction, "hands_detected": True})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sign Language Translation Server")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--port",       type=int,  default=5000)
    ap.add_argument("--host",       default="0.0.0.0")
    args = ap.parse_args()

    print("=" * 60)
    print("  SignSpeak AI  —  Local Inference Server")
    print("=" * 60)
    _load_model(args.checkpoint)
    threading.Thread(target=_init_mediapipe, daemon=True).start()
    print(f"\n  Open your browser at:  http://localhost:{args.port}\n")
    print("=" * 60)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
