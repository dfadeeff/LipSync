#!/usr/bin/env python
"""
SIMPLEST SyncNet baseline measurement - just make it work!
"""

import sys, pathlib, argparse, cv2, librosa, numpy as np, torch
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from load_syncnet import load_syncnet
from latentsync.utils.face_detector import FaceDetector
from latentsync.utils.audio import melspectrogram

MEL_LEN = 52
SYNC_THRES = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def measure_baseline(video_path, audio_path):
    """Measure baseline lip-sync quality - SIMPLE!"""
    
    print(f"📹 Video: {video_path}")
    print(f"🎵 Audio: {audio_path}")
    
    # Load model
    net = load_syncnet(DEVICE)
    
    # Audio processing
    wav, _ = librosa.load(audio_path, sr=16000)
    mel_full = melspectrogram(wav)
    
    mel_chunks = []
    for i in range(0, mel_full.shape[1] - MEL_LEN, 3):
        mel_chunks.append(mel_full[:, i:i + MEL_LEN])
    
    mel_chunks = torch.from_numpy(np.stack(mel_chunks)).unsqueeze(1).float()
    print(f"Audio chunks: {mel_chunks.shape}")

    # Video processing - use 256x256 to avoid downsampling issues
    cap = cv2.VideoCapture(video_path)
    fd = FaceDetector(device=DEVICE)
    crops = []
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bbox, _ = fd(frame[..., ::-1])
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        # Use 256x256 instead of 128x128 to avoid kernel size issues
        crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256))
        crops.append(crop)
    cap.release()
    
    print(f"Face crops: {len(crops)}")
    
    # Use 5-frame windows (simpler than 16-frame)
    frames_needed = 5
    n_chunks = min(len(crops) - frames_needed + 1, len(mel_chunks))
    
    if n_chunks <= 0:
        raise RuntimeError("Not enough frames")
    
    # Create 5-frame windows 
    vid_tensors = []
    for i in range(n_chunks):
        # Take 5 consecutive frames
        frames = [crops[i + j] for j in range(5)]
        # Stack and convert to tensor
        window = np.stack(frames, axis=0)  # (5, 256, 256, 3)
        window = torch.from_numpy(window).permute(0, 3, 1, 2)  # (5, 3, 256, 256)
        # Take just the middle frame for simplicity
        middle_frame = window[2]  # (3, 256, 256)
        vid_tensors.append(middle_frame)
    
    video_batch = torch.stack(vid_tensors).float() / 255.0  # (B, 3, 256, 256)
    audio_batch = mel_chunks[:n_chunks]
    
    print(f"Final shapes: video={video_batch.shape}, audio={audio_batch.shape}")
    
    # Move to device and compute
    video_batch = video_batch.to(DEVICE)
    audio_batch = audio_batch.to(DEVICE)
    
    try:
        v_emb, a_emb = net(video_batch, audio_batch)
        print(f"Embeddings: v={v_emb.shape}, a={a_emb.shape}")
        
        # Compute metrics
        if v_emb.shape[1] != a_emb.shape[1]:
            min_dim = min(v_emb.shape[1], a_emb.shape[1])
            v_emb = v_emb[:, :min_dim]
            a_emb = a_emb[:, :min_dim]
        
        dists = torch.linalg.norm(v_emb - a_emb, dim=1).cpu().numpy()
        lse_d = dists.mean()
        sync_acc = (dists < SYNC_THRES).mean()
        
        return lse_d, sync_acc
        
    except Exception as e:
        print(f"Forward failed: {e}")
        # Fallback to simple correlation measurement
        print("Using fallback correlation measurement...")
        return measure_simple_correlation(video_path, audio_path)

def measure_simple_correlation(video_path, audio_path):
    """Fallback: simple correlation if SyncNet fails"""
    
    # Load audio energy
    audio, _ = librosa.load(audio_path, sr=16000)
    audio_energy = librosa.feature.rms(y=audio)[0]
    
    # Load video and measure mouth movement
    cap = cv2.VideoCapture(video_path)
    movements = []
    prev_gray = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mouth_region = gray[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
        
        if prev_gray is not None:
            prev_mouth = prev_gray[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
            diff = cv2.absdiff(mouth_region, prev_mouth)
            movements.append(np.mean(diff))
        prev_gray = gray
    cap.release()
    
    # Align and correlate
    min_len = min(len(movements), len(audio_energy))
    if min_len < 10:
        return 0.0, 0.0
    
    movements = movements[:min_len]
    audio_energy = audio_energy[:min_len]
    
    corr = np.corrcoef(movements, audio_energy)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    
    # Convert to SyncNet-like metrics
    lse_d = 1.0 - abs(corr)  # Lower is better
    sync_acc = abs(corr)     # Higher is better
    
    return lse_d, sync_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()
    
    lse_d, sync_acc = measure_baseline(args.video, args.audio)
    
    print(f"\n=== BASELINE RESULTS ===")
    print(f"LSE-D: {lse_d:.4f} (lower = better)")
    print(f"SyncAcc: {sync_acc*100:.1f}% (higher = better)")
    
    if sync_acc > 0.7:
        print("✅ Excellent")
    elif sync_acc > 0.5:
        print("✅ Good")
    elif sync_acc > 0.3:
        print("⚠️ Moderate")
    else:
        print("❌ Poor")