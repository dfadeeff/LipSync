#!/usr/bin/env python3
"""
Profile LatentSync end-to-end (speed, memory, quality).
Usage example:
    python tools/profile_inference.py assets/demo1_video.mp4 assets/demo1_audio.wav \
           --steps 20 --scale 1.5
"""

import argparse, contextlib, sys
from pathlib import Path

import cv2, numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torch.profiler import (profile, record_function, ProfilerActivity,
                            tensorboard_trace_handler)

# ───────── repository imports ─────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.inference import main as run_inference          # your real pipeline
from gradio_app        import CONFIG_PATH, create_args, clear_gpu_memory
# ──────────────────────────────────────

LOG_DIR = Path("logs/latentsync_prof")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------#
# helpers                                                                      #
# -----------------------------------------------------------------------------#


def label(name):
    """record_function if CUDA is on, else no-op."""
    return record_function(name) if torch.cuda.is_available() else contextlib.nullcontext()

# -----------------------------------------------------------------------------#
# main profiler wrapper                                                        #
# -----------------------------------------------------------------------------#
def profiled_run(video: str, audio: str,
                 steps: int, scale: float, seed: int = 1247) -> None:

    # mimic your Gradio front-end
    cfg  = OmegaConf.load(CONFIG_PATH)
    cfg.run.inference_steps = steps
    cfg.run.guidance_scale  = scale

    out_dir = Path("temp");  out_dir.mkdir(exist_ok=True)
    out_mp4 = out_dir / f"{Path(video).stem}_profiled.mp4"
    args    = create_args(video, audio, str(out_mp4), steps, scale, seed)

    clear_gpu_memory()
    torch.cuda.reset_peak_memory_stats()

    # ------------------- profiler -------------------------------------------
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=tensorboard_trace_handler(str(LOG_DIR), use_gzip=True)
    )

    with prof:
        with label("00|full-pipeline"):
            run_inference(cfg, args)

    # ------------------- TOP-20 ---------------------------------------------
    print("\n═════ TOP-20 ops by self-CUDA time ═════")
    # print(prof.key_averages(group_by_input_shape=False)
    #           .table(sort_by="self_cuda_time_total", row_limit=20))
    events = prof.key_averages(group_by_input_shape=False)
    top = sorted(events, key=lambda e: getattr(e, "self_cuda_time_total", 0.0),
             reverse=True)[:20]
    
    # Pretty print – widths tweaked so it fits an 80-column terminal
    for ev in top:
        cuda_ms = ev.self_cuda_time_total / 1e3
        cpu_ms  = ev.self_cpu_time_total  / 1e3
        print(f"{ev.key[:40]:40s}  {cuda_ms:9.2f}  {cpu_ms:9.2f}  {ev.count:5d}")


    # ------------------- GPU memory -----------------------------------------
    if torch.cuda.is_available():
        print(f"\nPeak CUDA allocated : {torch.cuda.max_memory_allocated()/1e9:6.2f} GB")
        print(f"Peak CUDA reserved  : {torch.cuda.max_memory_reserved()/1e9:6.2f} GB")

    print(f"\nTrace saved under {LOG_DIR}  (open in TensorBoard → Profiler tab)")

    # ------------------- quick quality check --------------------------------
    try:
        def _quality(ref, gen, max_frames=100):
            cr, cg = cv2.VideoCapture(ref), cv2.VideoCapture(gen)
            n = int(min(cr.get(cv2.CAP_PROP_FRAME_COUNT),
                        cg.get(cv2.CAP_PROP_FRAME_COUNT), max_frames))
            psnrs, ssims = [], []
            for _ in range(n):
                ok_r, fr_r = cr.read(); ok_g, fr_g = cg.read()
                if not (ok_r and ok_g): break
                fr_g = cv2.resize(fr_g, (fr_r.shape[1], fr_r.shape[0]))
                psnrs.append(psnr(fr_r, fr_g, data_range=255))
                ssims.append(ssim(cv2.cvtColor(fr_r, cv2.COLOR_BGR2GRAY),
                                  cv2.cvtColor(fr_g, cv2.COLOR_BGR2GRAY)))
            cr.release(); cg.release()
            return float(np.mean(psnrs)), float(np.mean(ssims)), n

        p, s, n = _quality(video, str(out_mp4))
        print(f"\nQuality on {n} frames →  PSNR {p:5.2f} dB   SSIM {s:0.4f}")
    except Exception as e:
        print(f"\n[quality check skipped] {e}")

    print(f"\nOutput video → {out_mp4}\n")

# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("video");  cli.add_argument("audio")
    cli.add_argument("--steps", type=int,   default=20)
    cli.add_argument("--scale", type=float, default=1.5)
    cli.add_argument("--seed",  type=int,   default=1247)
    opt = cli.parse_args()

    if not Path(opt.video).exists():
        sys.exit(f"Video not found: {opt.video}")
    if not Path(opt.audio).exists():
        sys.exit(f"Audio not found: {opt.audio}")

    profiled_run(opt.video, opt.audio, opt.steps, opt.scale, opt.seed)