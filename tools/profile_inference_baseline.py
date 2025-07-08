#!/usr/bin/env python3
"""
Profile LatentSync end-to-end (speed, memory, quality).

Example:
    python tools/profile_inference.py assets/demo1_video.mp4 assets/demo1_audio.wav \
           --steps 20 --scale 1.5
"""

import argparse, contextlib, sys
from pathlib import Path
import cv2, numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity   as ssim
import torch
import functools
from omegaconf import OmegaConf
from torch.profiler import (
    profile, record_function, ProfilerActivity, schedule,
    tensorboard_trace_handler
)

# ───────── repository imports ─────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.inference import main as run_inference          # your real pipeline
from gradio_app        import CONFIG_PATH, create_args, clear_gpu_memory
# ──────────────────────────────────────

LOG_DIR = Path("logs/latentsync_prof")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def label(name):
    """record_function if CUDA is on, else no-op."""
    return record_function(name) if torch.cuda.is_available() else contextlib.nullcontext()

# ───────── one-shot profiler ─────────
def profiled_run(video: str, audio: str, steps: int, scale: float, seed: int = 1247) -> None:

    # recreate cfg & argparse.Namespace exactly like the Gradio front-end
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.run.inference_steps = steps
    cfg.run.guidance_scale  = scale

    out_dir = Path("temp"); out_dir.mkdir(exist_ok=True)
    out_mp4 = out_dir / f"{Path(video).stem}_profiled.mp4"
    args    = create_args(video, audio, str(out_mp4), steps, scale, seed)

    clear_gpu_memory()
    torch.cuda.reset_peak_memory_stats()

    # --- torch profiler ------------------------------------------------------
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=tensorboard_trace_handler(str(LOG_DIR), use_gzip=True)
)

    with prof:                                  #   step 0  →  ACTIVE
        with label("00|full-pipeline"):
            run_inference(cfg, args)


    # --- console summary -----------------------------------------------------
    print("\n═════ TOP-20 ops by CUDA time ═════")
    print(prof.key_averages(group_by_input_shape=False)
              .table(sort_by="self_cuda_time_total", row_limit=20))
    print("\n═════ HIGH-LEVEL rf() scopes (sorted by CUDA time) ═════")



    # keep only the events whose name contains our "|" tags
    scopes = prof.key_averages(group_by_input_shape=False)
    scopes = [e for e in scopes if "|" in e.key]      

    # pretty-print, aggregated by children kernels
    for evt in sorted(scopes,
                   key=lambda e: e.cuda_time_total,
                   reverse=True):
        # evt.cuda_time_total is in μs; convert to seconds
        cuda_s  = evt.cuda_time_total / 1e6
        cpu_s   = evt.cpu_time_total  / 1e6
        print(f"{evt.key:<25}  CUDA {cuda_s:8.3f}s   "
            f"CPU {cpu_s:8.3f}s   calls {evt.count:3d}")
        
        

    if torch.cuda.is_available():
        print(f"\nPeak CUDA allocated : {torch.cuda.max_memory_allocated() / 1e9:6.2f} GB")
        print(f"Peak CUDA reserved  : {torch.cuda.max_memory_reserved()  / 1e9:6.2f} GB")

    print(f"\nTrace saved under {LOG_DIR}  "
          f"(TensorBoard → Profiler tab, run dropdown).")

    # --- quick quality check -------------------------------------------------
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
        print(f"\n[quality skipped] {e}")

    print(f"\nOutput video → {out_mp4}\n")

# ───────── CLI wrapper ─────────
if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("video"), cli.add_argument("audio")
    cli.add_argument("--steps",  type=int,   default=20)
    cli.add_argument("--scale",  type=float, default=1.5)
    cli.add_argument("--seed",   type=int,   default=1247)
    opt = cli.parse_args()

    assert Path(opt.video ).exists(), f"Video not found: {opt.video}"
    assert Path(opt.audio ).exists(), f"Audio not found: {opt.audio}"

    profiled_run(opt.video, opt.audio, opt.steps, opt.scale, opt.seed)