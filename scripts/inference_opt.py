# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from omegaconf import OmegaConf


import torch
torch.backends.cuda.matmul.allow_tf32 = True
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper
from torch.profiler import record_function as rf
from torch.amp import autocast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")

    # Check if the GPU supports float16
    #is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    #dtype = torch.float16 if is_fp16_supported else torch.float32

    dtype = torch.float16

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    

    # ───────────────────────── COMPONENT 01 ─────────────────────────
    with rf("01|init-components"):
        scheduler = DDIMScheduler.from_pretrained("configs")

        if config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")

        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        vae.enable_tiling()
        vae.enable_slicing()
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            args.inference_ckpt_path,
            device="cpu",
        )

        # move it to GPU first (and pick FP16 if available)
        unet = unet.to(device="cuda", dtype=dtype)

        # enable xFormers attention
        unet.set_use_memory_efficient_attention_xformers(True) 


    # ───────────────────────── COMPONENT 02 ─────────────────────────
    with rf("02|build-pipeline"):
        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to("cuda")

    # use DeepCache

    # ───────────────────────── COMPONENT 03 ─────────────────────────
    with rf("03|deepcache-setup"):
        if args.enable_deepcache:
            helper = DeepCacheSDHelper(pipe=pipeline)
            helper.set_params(cache_interval=2, cache_branch_id=0)
            helper.enable()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")


    # ───────────────────────── COMPONENT 04 ─────────────────────────
    with rf("04|run-pipeline"), autocast(device_type="cuda", dtype=dtype):
        pipeline(
            video_path=args.video_path,
            audio_path=args.audio_path,
            video_out_path=args.video_out_path,
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir=args.temp_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)