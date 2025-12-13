#!/usr/bin/env python3
"""Multi-GPU driver for svg_generate.py.

This script splits the prompt list into disjoint ranges and launches one
svg_generate.py process per GPU, each restricted to its own index range.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

from svg_generate import load_metadata  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU wrapper for SVG-DiT generator")
    parser.add_argument("metadata_file", type=str, help="JSON or JSONL file containing prompts")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to store generated images")

    parser.add_argument("--n-samples", dest="n_samples", type=int, default=4, help="Images to generate per prompt")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--time-shifting-factor", type=float, default=10.0, help="Flow time shifting for DPM solver")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--precision", type=str, default="fp16", help="Precision passed through to svg_generate.py")
    parser.add_argument(
        "--solver",
        type=str,
        default="dpm",
        choices=["dpm", "midpoint", "euler", "heun"],
        help="Sampler backend (passed through)",
    )
    parser.add_argument(
        "--system-type",
        type=str,
        default="base",
        help="System prompt type (passed through)",
    )
    parser.add_argument("--skip-grid", action="store_true", help="Disable grid visualization")
    parser.add_argument("--grid-columns", type=int, default=2, help="Number of images per row in the grid")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt text")
    parser.add_argument("--ema", action="store_true", help="Use EMA weights if available")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for gated models")

    parser.add_argument(
        "--text-encoder",
        type=str,
        default=None,
        help="Path or HF repo id for the text encoder (overrides default in svg_generate.py if set)",
    )
    parser.add_argument(
        "--autoencoder-config",
        type=str,
        default=None,
        help="Path to SVG autoencoder config YAML (overrides default if set)",
    )
    parser.add_argument(
        "--dinov3-stats",
        type=str,
        default=None,
        help="Path to DinoV3 statistics tensor (overrides default if set)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        nargs="+",
        default=["512:512x512"],
        help="Resolution spec like '1024:720x1280'",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help='Comma-separated list of GPU ids to use, e.g. "0,1,2,3".',
    )

    return parser.parse_args()


def chunk_indices(n_total: int, n_chunks: int) -> List[range]:
    """Split [0, n_total) into n_chunks contiguous ranges."""
    base = n_total // n_chunks
    rem = n_total % n_chunks
    ranges: List[range] = []
    start = 0
    for i in range(n_chunks):
        length = base + (1 if i < rem else 0)
        end = start + length
        ranges.append(range(start, end))
        start = end
    return ranges


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_file)
    entries = load_metadata(metadata_path)
    if not entries:
        raise RuntimeError("No metadata entries found")

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if not gpu_ids:
        raise RuntimeError("No valid GPU ids provided via --gpus")

    n_total = len(entries)
    ranges = chunk_indices(n_total, len(gpu_ids))

    script_path = Path(__file__).resolve().with_name("svg_generate.py")
    procs: List[subprocess.Popen] = []

    for gpu_id, idx_range in zip(gpu_ids, ranges):
        if len(idx_range) == 0:
            continue
        start = idx_range.start
        end = idx_range.stop

        cmd = [
            "python",
            str(script_path),
            str(metadata_path),
            "--ckpt",
            args.ckpt,
            "--outdir",
            args.outdir,
            "--n-samples",
            str(args.n_samples),
            "--steps",
            str(args.steps),
            "--cfg-scale",
            str(args.cfg_scale),
            "--time-shifting-factor",
            str(args.time_shifting_factor),
            "--seed",
            str(args.seed),
            "--precision",
            args.precision,
            "--solver",
            args.solver,
            "--system-type",
            args.system_type,
            "--grid-columns",
            str(args.grid_columns),
            "--resolution",
            *args.resolution,
            "--start-index",
            str(start),
            "--end-index",
            str(end),
        ]

        if args.skip_grid:
            cmd.append("--skip-grid")
        if args.ema:
            cmd.append("--ema")
        if args.hf_token:
            cmd.extend(["--hf-token", args.hf_token])
        if args.negative_prompt:
            cmd.extend(["--negative-prompt", args.negative_prompt])
        if args.text_encoder:
            cmd.extend(["--text-encoder", args.text_encoder])
        if args.autoencoder_config:
            cmd.extend(["--autoencoder-config", args.autoencoder_config])
        if args.dinov3_stats:
            cmd.extend(["--dinov3-stats", args.dinov3_stats])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        print(f"[GPU {gpu_id}] Processing indices [{start}, {end}) with command: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)

    # Wait for all workers
    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more workers failed, exit codes: {exit_codes}")


if __name__ == "__main__":
    main()


