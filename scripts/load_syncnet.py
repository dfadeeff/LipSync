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
    """Return the list of out-channels for every conv1 in the down-blocks."""
    pat = re.compile(rf"^{prefix}\.(\d+)\.conv1\.weight$")
    chans = {int(m.group(1)): w.shape[0]
             for k, w in sd.items() if (m := pat.match(k))}
    if not chans:
        raise RuntimeError("Cannot find any down-block weights — bad prefix?")
    return [chans[i] for i in sorted(chans)]

def load_syncnet(device=DEVICE):
    raw = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd  = raw.get("state_dict", raw)                 # unwrap if needed

    audio  = scan_blocks(sd, "audio_encoder.down_blocks")
    visual = scan_blocks(sd, "visual_encoder.down_blocks")
    vin_ch = next(w.shape[1] for k, w in sd.items()
                  if k == "visual_encoder.conv_in.weight")

    cfg = dict(
        audio_encoder  = dict(in_channels=1,
            block_out_channels=audio,
            downsample_factors=[2]*len(audio),
            dropout=0.0, attn_blocks=[0]*len(audio)),
        visual_encoder = dict(in_channels=vin_ch,          # 48 for pixel space
            block_out_channels=visual,
            downsample_factors=[2]*len(visual),
            dropout=0.0, attn_blocks=[0]*len(visual)),
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