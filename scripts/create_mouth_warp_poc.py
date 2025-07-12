#!/usr/bin/env python
# scripts/create_mouth_warp_poc.py
"""
Advanced POC: Generates a new video by warping and blending the best-synced
mouth from a nearby frame onto the current frame's head pose.
This creates a video with both optimal lip-sync and smooth head motion.
"""

import sys, pathlib, argparse, cv2, librosa, numpy as np, torch, shutil, subprocess, dlib
from tqdm import tqdm

# ───────── Setup ──────────────────────────────────────────────────
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from load_syncnet import load_syncnet
from latentsync.utils.face_detector import FaceDetector
from latentsync.utils.audio import melspectrogram

# ───────── Constants ──────────────────────────────────────────────
SEARCH_WINDOW_SIZE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DLIB_LANDMARK_MODEL = "checkpoints/dlib/shape_predictor_68_face_landmarks.dat"
MOUTH_LANDMARK_INDICES = list(range(48, 68))

# ───────── Helper Functions ───────────────────────────────────────
def get_landmarks(image, face_rect, landmark_predictor):
    """Detects facial landmarks within a given face rectangle."""
    detection_object = landmark_predictor(image, face_rect)
    
    # --- START OF THE FIX ---
    # The landmark_predictor returns a 'full_object_detection' object.
    # We must call the .parts() method on it to get the iterable list of points.
    points = detection_object.parts()
    return np.array([(p.x, p.y) for p in points], dtype=np.int32)
    # --- END OF THE FIX ---

def warp_and_blend_mouth(target_frame, source_frame, landmarks_target, landmarks_source):
    """
    Extracts the mouth from the source using a precise, FEATHERED mask,
    warps it to fit the target, and seamlessly blends it.
    """
    source_mouth_points = landmarks_source[MOUTH_LANDMARK_INDICES]
    target_mouth_points = landmarks_target[MOUTH_LANDMARK_INDICES]

    # --- START OF THE NEW IMPROVEMENT (FEATHERED MASK) ---
    
    # 1. Create a tight hull around the lips for the core of the mask
    source_mouth_hull = cv2.convexHull(source_mouth_points)
    
    # 2. Create a slightly larger, softer hull for the blend region
    # We can do this by finding the center of the mouth and scaling the points out
    center = np.mean(source_mouth_points, axis=0)
    scaled_points = (source_mouth_points - center) * 1.15 + center
    source_outer_hull = cv2.convexHull(scaled_points.astype(np.int32))

    # 3. Create the mask with a feathered edge
    # Draw the outer hull in a mid-gray color
    source_mask = np.zeros(source_frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(source_mask, source_outer_hull, 128)
    # Draw the inner, tight hull in pure white
    cv2.fillConvexPoly(source_mask, source_mouth_hull, 255)
    # Apply a strong blur to create a smooth gradient between the regions
    source_mask = cv2.GaussianBlur(source_mask, (15, 15), 0)

    # --- END OF THE NEW IMPROVEMENT ---
    
    # Calculate the transformation to warp the source mouth to the target's position
    h, _ = cv2.findHomography(source_mouth_points, target_mouth_points, cv2.RANSAC)
    if h is None:
        return target_frame

    # Get the bounding box of the target mouth for the blend center
    (x, y, w, h_rect) = cv2.boundingRect(target_mouth_points)
    
    # Warp the source frame using the calculated transformation
    warped_source_frame = cv2.warpPerspective(source_frame, h, (target_frame.shape[1], target_frame.shape[0]))
    
    # Warp the feathered source mask to the target's perspective
    warped_mask = cv2.warpPerspective(source_mask, h, (target_frame.shape[1], target_frame.shape[0]))

    # Use seamless cloning to blend the result
    center = (x + w // 2, y + h_rect // 2)
    output_frame = cv2.seamlessClone(warped_source_frame, target_frame, warped_mask, center, cv2.NORMAL_CLONE)
    
    return output_frame

@torch.no_grad()
def generate_warped_video(video_path: str, audio_path: str, output_path: str):
    print("--- Advanced POC: Generative Mouth Warping and Blending ---")

    if not pathlib.Path(DLIB_LANDMARK_MODEL).exists():
        raise FileNotFoundError(f"Dlib landmark model not found at {DLIB_LANDMARK_MODEL}. Please run the download instructions.")

    # A. Initial Setup
    temp_frame_dir = pathlib.Path(output_path).parent / "temp_frames_warp"
    if temp_frame_dir.exists(): shutil.rmtree(temp_frame_dir)
    temp_frame_dir.mkdir(parents=True)

    net = load_syncnet(DEVICE).eval()
    fd = FaceDetector(device=DEVICE)
    landmark_predictor = dlib.shape_predictor(DLIB_LANDMARK_MODEL)

    # B. Load Audio & Video Data
    print("Step 1/5: Processing audio and video frames...")
    wav, _ = librosa.load(audio_path, sr=16000)
    mel = melspectrogram(wav)
    audio_chunks = [mel[:, i:i+16] for i in range(0, mel.shape[1] - 16, 3)]
    audio_chunks_tensor = torch.from_numpy(np.stack(audio_chunks)).unsqueeze(1).float().to(DEVICE)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_data = []
    last_box = None
    frame_idx_counter = 0
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Detecting faces/landmarks") as pbar:
        while True:
            ok, frm = cap.read()
            if not ok: break
            
            box, _ = fd(frm[..., ::-1])
            box = box if box is not None else last_box
            if box is not None:
                last_box = box
                x1, y1, x2, y2 = box
                face_crop = cv2.resize(frm[y1:y2, x1:x2], (256, 256))
                dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
                landmarks = get_landmarks(frm, dlib_rect, landmark_predictor)
                frame_data.append({"frame": frm, "crop": face_crop, "landmarks": landmarks, "original_index": frame_idx_counter})
            else:
                frame_data.append(None)
            frame_idx_counter += 1
            pbar.update(1)
    cap.release()

    if len(frame_data) < 16: raise RuntimeError("Video too short.")
    
    # C. Calculate Embeddings & Find Best Matches
    print("Step 2/5: Calculating embeddings...")
    audio_embeds = net.audio_proj(net.audio_encoder(audio_chunks_tensor).reshape(len(audio_chunks_tensor), -1))

    face_crops = [d['crop'] for d in frame_data if d is not None]
    video_chunk_embeds = []
    num_video_chunks = len(face_crops) - 15
    for i in tqdm(range(num_video_chunks), desc="Getting video embeddings"):
        window = np.stack(face_crops[i : i + 16])
        clip = torch.from_numpy(window).permute(0, 3, 1, 2).reshape(1, -1, 256, 256).float().div_(255.).to(DEVICE)
        v_emb = net.visual_proj(net.visual_encoder(clip).reshape(1, -1))
        video_chunk_embeds.append(v_emb)
    video_embeds = torch.cat(video_chunk_embeds)
    
    num_chunks = min(len(audio_embeds), len(video_embeds))
    
    print("Step 3/5: Finding best mouth matches for each frame...")
    best_match_indices = []
    for i in tqdm(range(num_chunks), desc="Finding best matches"):
        start = max(0, i - SEARCH_WINDOW_SIZE)
        end = min(num_chunks, i + SEARCH_WINDOW_SIZE + 1)
        distances = torch.linalg.norm(video_embeds[start:end] - audio_embeds[i], dim=1)
        best_idx = start + torch.argmin(distances)
        best_match_indices.append(best_idx)

    # D. Generate New Frames via Warping and Blending
    print("Step 4/5: Generating new frames with warped mouths...")
    for i in tqdm(range(num_chunks), desc="Compositing frames"):
        target_data = frame_data[i]
        source_idx = best_match_indices[i]
        source_data = frame_data[source_idx]

        if target_data is None or source_data is None:
            final_frame = frame_data[i]['frame'] if target_data and 'frame' in target_data else np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
        else:
            final_frame = warp_and_blend_mouth(
                target_data['frame'], 
                source_data['frame'],
                target_data['landmarks'],
                source_data['landmarks']
            )
        
        frame_path = temp_frame_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(frame_path), final_frame)

    # E. Stitch Final Video with Audio
    print(f"Step 5/5: Stitching frames into final video at {output_path}...")
    silent_video_path = temp_frame_dir / "silent.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", f"{temp_frame_dir}/frame_%06d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent_video_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    merge_cmd = [
        "ffmpeg", "-y", "-i", str(silent_video_path), "-i", audio_path, 
        "-c:v", "copy", "-c:a", "aac", str(pathlib.Path(output_path))
    ]
    subprocess.run(merge_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    shutil.rmtree(temp_frame_dir)
    print("--- Advanced POC video generation complete. ---")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generates a new video via generative mouth warping.")
    p.add_argument("--video", required=True, help="Original input video.")
    p.add_argument("--audio", required=True, help="Corresponding audio file.")
    p.add_argument("--output", required=True, help="Path to save the FINAL generated video.")
    args = p.parse_args()
    generate_warped_video(args.video, args.audio, args.output)