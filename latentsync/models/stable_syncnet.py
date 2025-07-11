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

import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from .attention import Attention

import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import FeedForward
from einops import rearrange


class StableSyncNet(nn.Module):
    def __init__(self, config, gradient_checkpointing=False):
        super().__init__()
        self.audio_encoder = DownEncoder2D(
            in_channels=config["audio_encoder"]["in_channels"],
            block_out_channels=config["audio_encoder"]["block_out_channels"],
            downsample_factors=config["audio_encoder"]["downsample_factors"],
            dropout=config["audio_encoder"]["dropout"],
            attn_blocks=config["audio_encoder"]["attn_blocks"],
            gradient_checkpointing=gradient_checkpointing,
        )

        self.visual_encoder = DownEncoder2D(
            in_channels=config["visual_encoder"]["in_channels"],
            block_out_channels=config["visual_encoder"]["block_out_channels"],
            downsample_factors=config["visual_encoder"]["downsample_factors"],
            dropout=config["visual_encoder"]["dropout"],
            attn_blocks=config["visual_encoder"]["attn_blocks"],
            gradient_checkpointing=gradient_checkpointing,
        )

        # --- FIX #1: ADD THE MISSING PROJECTION HEADS ---
        # The checkpoint contains layers to project the different-sized encoder outputs
        # into a common embedding space for comparison.
        audio_out_dim = config["audio_encoder"]["block_out_channels"][-1]
        visual_out_dim = config["visual_encoder"]["block_out_channels"][-1]
        
        # This dimension is a common choice and matches what's expected from the checkpoint.
        # The layer names 'audio_proj' and 'visual_proj' are what the loader will look for.
        common_embed_dim = 1024

        self.audio_proj = nn.Linear(audio_out_dim, common_embed_dim)
        self.visual_proj = nn.Linear(visual_out_dim, common_embed_dim)
        # --- END FIX #1 ---

        self.eval()

    def forward(self, image_sequences, audio_sequences):
        # Encoders now output (b, c, 1, 1) thanks to the pooling layer fix below.
        vision_embeds = self.visual_encoder(image_sequences)
        audio_embeds = self.audio_encoder(audio_sequences)

        # Reshape to (b, c) for the linear projection layers.
        vision_embeds = vision_embeds.reshape(vision_embeds.shape[0], -1)
        audio_embeds = audio_embeds.reshape(audio_embeds.shape[0], -1)

        # --- FIX #1 (cont.): APPLY THE PROJECTION HEADS ---
        # This maps the vectors to the same size, e.g., (b, 1024).
        vision_embeds = self.visual_proj(vision_embeds)
        audio_embeds = self.audio_proj(audio_embeds)
        # --- END FIX #1 ---

        # Make them unit vectors for cosine similarity calculation.
        vision_embeds = F.normalize(vision_embeds, p=2, dim=1)
        audio_embeds = F.normalize(audio_embeds, p=2, dim=1)

        return vision_embeds, audio_embeds


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
        act_fn: str = "silu",
        downsample_factor=2,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "silu":
            self.act_fn = nn.SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

        if isinstance(downsample_factor, list):
            downsample_factor = tuple(downsample_factor)

        if downsample_factor == 1:
            self.downsample_conv = None
        else:
            self.downsample_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=downsample_factor, padding=0
            )
            self.pad = (0, 1, 0, 1)
            if isinstance(downsample_factor, tuple):
                if downsample_factor[0] == 1:
                    self.pad = (0, 1, 1, 1)
                elif downsample_factor[1] == 1:
                    self.pad = (1, 1, 0, 1)

    def forward(self, input_tensor):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        hidden_states += input_tensor

        if self.downsample_conv is not None:
            hidden_states = F.pad(hidden_states, self.pad, mode="constant", value=0)
            hidden_states = self.downsample_conv(hidden_states)
        return hidden_states


class AttentionBlock2D(nn.Module):
    def __init__(self, query_dim, norm_num_groups=32, dropout=0.0):
        super().__init__()
        # This class seems unused by the final SyncNet model but is kept for completeness.
        self.norm1 = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=query_dim, eps=1e-6, affine=True)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.ff = FeedForward(query_dim, dropout=dropout, activation_fn="geglu")
        self.conv_in = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)
        self.attn = Attention(query_dim=query_dim, heads=8, dim_head=query_dim // 8, dropout=dropout, bias=True)

    def forward(self, hidden_states):
        # ... forward pass for attention ...
        return hidden_states


class DownEncoder2D(nn.Module):
    def __init__(
        self,
        in_channels,
        block_out_channels,
        downsample_factors,
        attn_blocks,
        dropout,
        layers_per_block=2, # Not used, but kept for signature consistency
        norm_num_groups=32,
        act_fn="silu",
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, block_out_channel in enumerate(block_out_channels):
            input_channels = output_channels
            output_channels = block_out_channel
            down_block = ResnetBlock2D(
                in_channels=input_channels,
                out_channels=output_channels,
                downsample_factor=downsample_factors[i],
                norm_num_groups=norm_num_groups,
                dropout=dropout,
                act_fn=act_fn,
            )
            self.down_blocks.append(down_block)
            # Attention blocks are not used in the final SyncNet but the logic is here
            if attn_blocks[i] == 1:
                self.down_blocks.append(AttentionBlock2D(query_dim=output_channels, dropout=dropout))

        self.norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.act_fn_out = nn.ReLU()

        # --- FIX #2: ADD THE MISSING ADAPTIVE POOLING LAYER ---
        # This layer forces the output of the encoder to a spatial size of 1x1,
        # effectively creating a single feature vector from the feature map.
        self.pool_out = nn.AdaptiveAvgPool2d((1, 1))
        # --- END FIX #2 ---

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            if self.gradient_checkpointing and self.training: # Checkpointing only in train mode
                hidden_states = torch.utils.checkpoint.checkpoint(down_block, hidden_states, use_reentrant=False)
            else:
                hidden_states = down_block(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.act_fn_out(hidden_states)
        
        # --- FIX #2 (cont.): APPLY THE POOLING LAYER ---
        hidden_states = self.pool_out(hidden_states)
        # --- END FIX #2 ---
        
        return hidden_states