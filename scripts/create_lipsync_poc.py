#!/usr/bin/env python
# scripts/create_lipsync_poc.py
"""
POC that generates a new, re-timed video to improve lip-sync.
This version uses a robust frame-saving and ffmpeg-stitching method
to guarantee video output, bypassing OpenCV's VideoWriter issues.
"""

import sys, pathlib, argparse, cv2, librosa, numpy as np, torch, shutil, subprocess
from tqdm import tqdm

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from load_syncnet                   import load_syncnet
from latentsync.utils.face_detector import FaceDetector
from latentsync.utils.audio         import melspectrogram

SEARCH_WINDOW_SIZE = 3
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def generate_improved_video(video_path: str, audio_path: str, output_path: str):
    print("--- Lip-Sync Improvement POC: Optimal Frame Re-Timing (Robust v3) ---")
    
    # Create a temporary directory to store individual frames
    temp_frame_dir = pathlib.Path(output_path).parent / "temp_frames"
    if temp_frame_dir.exists():
        shutil.rmtree(temp_frame_dir)
    temp_frame_dir.mkdir(parents=True)
    print(f"Using temporary directory for frames: {temp_frame_dir}")

    net = load_syncnet(DEVICE).eval()
    
    print("Step 1/5: Processing audio track...")
    # ... [Audio processing code remains the same] ...
    wav, _ = librosa.load(audio_path, sr=16000)
    mel = melspectrogram(wav)
    audio_chunks = [mel[:, i:i+16] for i in range(0, mel.shape[1] - 16, 3)]
    audio_chunks_tensor = torch.from_numpy(np.stack(audio_chunks)).unsqueeze(1).float().to(DEVICE)

    print("Step 2/5: Processing video frames...")
    fd = FaceDetector(device=DEVICE)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_frames, face_crops, last_box = [], [], None
    while True:
        ok, frm = cap.read()
        if not ok: break
        video_frames.append(frm)
        box, _ = fd(frm[..., ::-1])
        box = box if box is not None else last_box
        if box is not None:
            last_box = box
            x1, y1, x2, y2 = box
            face_crops.append(cv2.resize(frm[y1:y2, x1:x2], (256, 256)))
        else:
            face_crops.append(np.zeros((256, 256, 3), dtype=np.uint8))
    cap.release()

    if not video_frames or len(video_frames) < 16:
        raise RuntimeError("Video is too short.")

    print("Step 3/5: Finding best frame matches...")
    # ... [Embedding and matching logic remains the same] ...
    audio_embeds = net.audio_proj(net.audio_encoder(audio_chunks_tensor).reshape(len(audio_chunks_tensor), -1))

    video_chunk_embeds = []
    num_video_chunks = len(face_crops) - 15
    for i in tqdm(range(num_video_chunks), desc="Getting video embeddings"):
        window = np.stack(face_crops[i : i + 16])
        clip = torch.from_numpy(window).permute(0, 3, 1, 2).reshape(1, -1, 256, 256).float().div_(255.).to(DEVICE)
        v_emb = net.visual_proj(net.visual_encoder(clip).reshape(1, -1))
        video_chunk_embeds.append(v_emb)
    video_embeds = torch.cat(video_chunk_embeds)
    
    num_chunks = min(len(audio_embeds), len(video_embeds))
    new_frame_indices = []
    for i in tqdm(range(num_chunks), desc="Finding best matches"):
        start = max(0, i - SEARCH_WINDOW_SIZE)
        end = min(num_chunks, i + SEARCH_WINDOW_SIZE + 1)
        distances = torch.linalg.norm(video_embeds[start:end] - audio_embeds[i], dim=1)
        best_idx = start + torch.argmin(distances)
        new_frame_indices.append(best_idx)

    print(f"Step 4/5: Saving individual frames...")
    for i, frame_idx in enumerate(tqdm(new_frame_indices, desc="Saving frames")):
        frame_path = temp_frame_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(frame_path), video_frames[frame_idx])

    print(f"Step 5/5: Stitching frames into video with ffmpeg...")
    
    # --- START OF THE FIX ---
    # Use the proven ffmpeg method to create the final video with audio.
    # This completely bypasses the broken cv2.VideoWriter.
    silent_video_path = temp_frame_dir / "silent.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), 
        "-i", f"{temp_frame_dir}/frame_%06d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent_video_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Now merge the silent video with the original audio
    final_output_path = pathlib.Path(output_path)
    merge_cmd = [
        "ffmpeg", "-y", "-i", str(silent_video_path), 
        "-i", audio_path, 
        "-c:v", "copy", "-c:a", "aac", str(final_output_path)
    ]
    subprocess.run(merge_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # --- END OF THE FIX ---
    
    # Clean up the temporary frame directory
    shutil.rmtree(temp_frame_dir)
    
    print(f"--- Video generation complete. File saved to: {final_output_path} ---")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generates a new, re-timed video with improved lip-sync.")
    p.add_argument("--video", required=True, help="Original input video.")
    p.add_argument("--audio", required=True, help="Corresponding audio file.")
    p.add_argument("--output", required=True, help="Path to save the FINAL generated video with audio.")
    args = p.parse_args()
    generate_improved_video(args.video, args.audio, args.output)