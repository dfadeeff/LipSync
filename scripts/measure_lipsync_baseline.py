#!/usr/bin/env python
"""
Quick-&-light lip-sync quality heuristic – **no checkpoints, CPU-only**.

Metric  =  best Pearson-r between
           ▸ z-scored audio RMS envelope
           ▸ z-scored, median-filtered mouth-motion
           evaluated over temporal shifts -2 … +2 frames.

Outputs a 0-100 “sync score” (100 = perfect positive corr).
"""

import cv2, librosa, numpy as np, argparse
from scipy.signal import medfilt   # tiny, pure-python Median filter

# ──────────────────────────────────────────────────────────────────────────
def rms_envelope(wav, sr, hop=512, win=1024):
    return librosa.feature.rms(y=wav, frame_length=win, hop_length=hop)[0]

def mouth_motion(frames):
    """naïve grey-diff over centre-bottom ROI → 1-D motion signal"""
    mot, prev = [], None
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        roi = g[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
        if prev is not None:
            mot.append(np.mean(cv2.absdiff(roi, prev)))
        prev = roi
    return np.array(mot, dtype=np.float32)

def best_corr(a, b, max_shift=2):
    def z(v): return (v - v.mean()) / (v.std() + 1e-6)
    a, b = z(a), z(b)
    best = -1.0
    for s in range(-max_shift, max_shift+1):
        if s < 0:
            r = np.corrcoef(a[:s], b[-s:])[0,1]
        elif s > 0:
            r = np.corrcoef(a[s:], b[:-s])[0,1]
        else:
            r = np.corrcoef(a, b)[0,1]
        best = max(best, abs(r))        # use absolute corr to ignore sign flips
    return max(best, 0.0)               # NaN → 0

# ──────────────────────────────────────────────────────────────────────────
def measure(video, audio):
    print(f"Video  : {video}\nAudio  : {audio}")

    wav, sr = librosa.load(audio, sr=16000)
    a_env   = rms_envelope(wav, sr)

    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()

    if len(frames) < 6:
        raise RuntimeError("Need ≥6 frames for motion estimation")

    v_mot = mouth_motion(frames)
    v_mot = medfilt(v_mot, kernel_size=5)          # kill jitter

    L = min(len(a_env), len(v_mot))
    if L < 10:
        raise RuntimeError("Overlap too short for correlation")

    score = best_corr(a_env[:L], v_mot[:L]) * 100
    print(f"Processed {L} aligned frames")
    print(f"Sync score: {score:.1f}%")

    if score > 50:
        print("✅   Looks well-synced")
    elif score > 20:
        print("⚠️   Acceptable / borderline")
    else:
        print("❌   Poor sync")
    return score

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input MP4")
    ap.add_argument("--audio", required=True, help="matching WAV/MP3")
    args = ap.parse_args()
    measure(args.video, args.audio)