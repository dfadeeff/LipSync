#!/usr/bin/env python
# scripts/measure_lipsync.py
"""
Stable-SyncNet baseline scorer.
THIS VERSION INCLUDES A POC FOR ROBUST FACE TRACKING.
--------------------------------------------
Outputs
  • LSE-D   – lower is better
  • SyncAcc – higher is better
"""

import sys, pathlib, argparse, cv2, librosa, numpy as np, torch
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from load_syncnet                   import load_syncnet
from latentsync.utils.face_detector import FaceDetector
from latentsync.utils.audio         import melspectrogram

# ───────── constants ────────────────────────────────────────────────
FRAMES_PER_CHUNK = 16
MEL_LEN          = 16
MEL_HOP_FRAMES   = 3
SYNC_THRES       = 0.5
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ───────── helpers ──────────────────────────────────────────────────
def make_mel_chunks(audio_path:str):
    wav, _ = librosa.load(audio_path, sr=16000)
    mel    = melspectrogram(wav)
    chunks = [
        mel[:, i : i + MEL_LEN]
        for i in range(0, mel.shape[1] - MEL_LEN, MEL_HOP_FRAMES)
    ]
    if len(chunks) < 4:
        raise RuntimeError("Audio is too short for SyncNet.")
    return torch.from_numpy(np.stack(chunks)).unsqueeze(1).float()

def make_video_tensor(video_path:str, max_clips:int):
    fd  = FaceDetector(device=DEVICE)
    cap = cv2.VideoCapture(video_path)
    buf = []

    # --- POC IMPLEMENTATION START ---
    # We will store the last known location of the face.
    last_known_box = None
    print("Processing video with Robust Face Tracking POC...")
    # --- POC IMPLEMENTATION END ---

    while True:
        ok, frm = cap.read()
        if not ok:
            break
        
        box, _ = fd(frm[..., ::-1])

        # --- POC IMPLEMENTATION START ---
        # If the detector fails on this frame, use the last known box as a fallback.
        if box is None and last_known_box is not None:
            # print("Face detection failed. Using last known box.") # Uncomment for debugging
            box = last_known_box
        
        # If there's still no box (meaning no face has been detected yet),
        # then we have no choice but to skip.
        if box is None:
            # print("Skipping frame, no face has ever been detected.") # Uncomment for debugging
            continue

        # If we have a box (either new or carried-over), update the last known box.
        last_known_box = box
        # --- POC IMPLEMENTATION END ---
        
        x1, y1, x2, y2 = box
        crop = cv2.resize(frm[y1:y2, x1:x2], (256, 256))
        buf.append(crop)
    cap.release()

    if len(buf) < FRAMES_PER_CHUNK:
        raise RuntimeError("Too few detected faces to form a full chunk.")

    N = min(len(buf) - FRAMES_PER_CHUNK + 1, max_clips)
    clips = []
    for i in range(N):
        window = np.stack(buf[i : i + FRAMES_PER_CHUNK], axis=0)
        window = torch.from_numpy(window).permute(0, 3, 1, 2)
        clips.append(window.reshape(-1, 256, 256))
    return torch.stack(clips).float().div_(255.)

# ───────── core metric ──────────────────────────────────────────────
@torch.no_grad()
def measure_baseline(video:str, audio:str):
    print(f"📹 {video}\n🎵 {audio}")
    a = make_mel_chunks(audio)
    v = make_video_tensor(video, a.size(0))
    min_len = min(a.size(0), v.size(0))
    a = a[:min_len]
    v = v[:min_len]
    print("Shapes → video", tuple(v.shape), "audio", tuple(a.shape))
    net = load_syncnet(DEVICE)
    v_emb, a_emb = net(v.to(DEVICE), a.to(DEVICE))
    dist = torch.linalg.norm(v_emb - a_emb, dim=1).cpu().numpy()
    lse_d = float(dist.mean())
    acc   = float((dist < SYNC_THRES).mean())
    return lse_d, acc

# ───────── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--audio", required=True)
    args = p.parse_args()

    d, acc = measure_baseline(args.video, args.audio)
    print(f"\nLSE-D  : {d:.3f}   (↓ better)")
    print(f"SyncAcc: {acc*100:.1f}%   (↑ better)")