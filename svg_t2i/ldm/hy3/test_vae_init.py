#!/usr/bin/env python3
"""
GPU decoder-only test script for AutoencoderKLConv3D.

Initializes AutoencoderKLConv3D from vae_config.json,
then runs a forward test *only for the decoder* on GPU.
"""
import json
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# from autoencoder_kl_3d import AutoencoderKLConv3D
from autoencoder_kl_3d import Decoder


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main():
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    cfg_path = HERE / "vae_config.json"
    assert cfg_path.exists(), f"Config not found: {cfg_path}"

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    params = {
        # "in_channels": cfg["in_channels"],
        "out_channels": cfg["out_channels"],
        "z_channels": cfg["z_channels"],
        "block_out_channels": tuple(cfg["block_out_channels"]),
        "num_res_blocks": cfg["num_res_blocks"],
        "ffactor_spatial": cfg["ffactor_spatial"],
        "ffactor_temporal": cfg["ffactor_temporal"],
        # "sample_size": cfg.get("sample_size"),
        # "sample_tsize": cfg.get("sample_tsize"),
        # "scaling_factor": cfg.get("scaling_factor", None),
        # "downsample_match_channel": cfg.get("downsample_match_channel", True),
        "upsample_match_channel": cfg.get("upsample_match_channel", True),
    }

    print("\nInitializing AutoencoderKLConv3D (for decoder test only):")
    print({
        "out_channels": params["out_channels"],
        "z_channels": params["z_channels"],
        "block_out_channels": params["block_out_channels"],
        "ffactor_spatial": params["ffactor_spatial"],
        "ffactor_temporal": params["ffactor_temporal"],
    })

    decoder = Decoder(**params).to(device)
    decoder.eval()

    # Print param counts
    total_params = count_params(decoder)
    print(f"\nParameter counts:")
    print(f"  Total:   {total_params/1e6:.2f} M")

    # print(f'Decoder = {decoder}')
    
    # -----------------------------
    # Latent tensor for decoder test
    # -----------------------------
    latent_ch = params["z_channels"]
    ff_sp = params["ffactor_spatial"]
    ff_tp = params["ffactor_temporal"]

    # Target reconstructed size (e.g. 1024x1024x1)
    target_size = 1024
    target_t = 1

    latent_h = target_size // ff_sp
    latent_w = target_size // ff_sp
    latent_t = max(1, target_t // ff_tp)

    print(f"\nGenerating random latent input:")
    print(f"  latent shape = [1, {latent_ch}, {latent_t}, {latent_h}, {latent_w}]")

    z = torch.randn(1, latent_ch, latent_t, latent_h, latent_w, device=device)

    # -----------------------------
    # Decoder forward
    # -----------------------------
    with torch.no_grad():
        decoded = decoder(z)

        print(f"\nDecoder output:")
        print(f"  shape: {tuple(decoded.shape)}")
        print(f"  dtype: {decoded.dtype}")
        print(f"  min={decoded.min().item():.6f}, max={decoded.max().item():.6f}, mean={decoded.mean().item():.6f}")

    print("\n✅ Decoder-only GPU test complete.")


if __name__ == "__main__":
    main()