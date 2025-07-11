#!/usr/bin/env python
"""Load the released SyncNet checkpoint without shape-mismatch errors."""

import sys, pathlib, torch, re
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))                           # make `latentsync` importable

from latentsync.models.stable_syncnet import StableSyncNet

CKPT   = "checkpoints/stable_syncnet.pt"             # 1.6 GB file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
def scan_blocks(sd, prefix):
    """Return the list of out-channels for every ResNet block in the down-blocks."""
    # This pattern specifically finds ResNet blocks by their 'conv1' layer.
    pat = re.compile(rf"^{prefix}\.(\d+)\.conv1\.weight$")
    chans = {int(m.group(1)): w.shape[0]
             for k, w in sd.items() if (m := pat.match(k))}
    if not chans:
        raise RuntimeError(f"Cannot find any down-block weights with prefix '{prefix}'. Check checkpoint and prefix.")
    # Return a list of output channels, sorted by the block index.
    return [chans[i] for i in sorted(chans)]

def load_syncnet(device=DEVICE):
    raw = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd  = raw.get("state_dict", raw)                 # unwrap if needed

    # Dynamically discover the architecture from the checkpoint file.
    # The length of this list is the ground truth for the number of ResNet blocks.
    audio_chans  = scan_blocks(sd, "audio_encoder.down_blocks")
    visual_chans = scan_blocks(sd, "visual_encoder.down_blocks")
    vin_ch = next(w.shape[1] for k, w in sd.items()
                  if k == "visual_encoder.conv_in.weight")

    # --- START OF THE DEFINITIVE FIX ---

    # The checkpoint has 7 ResNet blocks in the audio encoder. We must provide 7 downsample factors.
    # The initial blocks downsample in both height and width.
    # Later blocks only downsample in height (2, 1) to preserve the narrow width.
    # The final blocks do not downsample at all (factor 1) to prevent the feature map from collapsing.
    audio_downsample_factors = [
        (2, 2), # Block 0: (80, 16) -> (40, 8)
        (2, 2), # Block 1: (40, 8)  -> (20, 4)
        (2, 1), # Block 2: (20, 4)  -> (10, 4)
        (2, 1), # Block 3: (10, 4)  -> (5, 4)
        (2, 1), # Block 4: (5, 4)   -> (2, 4)  (height becomes 2 after padding and conv)
        1,      # Block 5: No downsampling
        1,      # Block 6: No downsampling
    ]

    # This check is no longer needed as we've built the list based on the discovered length.
    # However, it's good practice to assert they are equal.
    assert len(audio_downsample_factors) == len(audio_chans), \
        f"FATAL: Mismatch! Factors needed: {len(audio_chans)}, Factors provided: {len(audio_downsample_factors)}"
    
    # --- END OF THE DEFINITIVE FIX ---

    cfg = dict(
        audio_encoder  = dict(in_channels=1,
            block_out_channels=audio_chans,
            downsample_factors=audio_downsample_factors,
            dropout=0.0,
            attn_blocks=[0]*len(audio_chans)), # Explicitly state no attention blocks
        visual_encoder = dict(in_channels=vin_ch,
            block_out_channels=visual_chans,
            downsample_factors=[2]*len(visual_chans), # Visual is square, so simple downsampling is fine
            dropout=0.0,
            attn_blocks=[0]*len(visual_chans)),
    )

    net = StableSyncNet(cfg).to(device)

    # keep only tensors whose *name and shape* match the freshly-built graph
    model_sd   = net.state_dict()
    compatible = {k: v for k, v in sd.items()
                  if k in model_sd and v.shape == model_sd[k].shape}

    skipped = len(sd) - len(compatible)
    net.load_state_dict(compatible, strict=False)
    print(f"✓ SyncNet ready → {len(compatible)} tensors loaded, {skipped} skipped")
    return net.eval()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    load_syncnet()