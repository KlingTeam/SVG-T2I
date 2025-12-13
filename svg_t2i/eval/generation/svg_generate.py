#!/usr/bin/env python3
"""Generate images with the SVG-DiT model for Geneval benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM  # type: ignore[attr-defined]
from omegaconf import OmegaConf

# --- Repository-local imports -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SVG_DIT_ROOT = PROJECT_ROOT / "svg_dit_t2i"
if str(SVG_DIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SVG_DIT_ROOT))

import models  # type: ignore  # noqa: E402
from transport import Sampler, create_transport  # type: ignore  # noqa: E402
from util import instantiate_from_config  # type: ignore  # noqa: E402
from sample_svg_t2i import (  # type: ignore  # noqa: E402
    encode_prompt,
    parse_ode_args,
    parse_transport_args,
    generate_enhanced_prompt
)


# -----------------------------------------------------------------------------
torch.set_grad_enabled(False)

# Provide a minimal torch._dynamo stub for older torch versions (<2.0).
if not hasattr(torch, "_dynamo"):
    class _NoOpDynamo:
        """Fallback shim so newer libs that expect torch._dynamo still run."""

        def mark_static_address(self, *_args, **_kwargs):
            return None

        def __getattr__(self, _name):
            # Return self for any other attribute access (best-effort no-op).
            return self

    torch._dynamo = _NoOpDynamo()  # type: ignore[attr-defined]

DEFAULT_TEXT_ENCODER= PROJECT_ROOT / "svg_t2i/pretrained_models/gemma-2-2b"
DEFAULT_AUTOENCODER_CONFIG = (
    PROJECT_ROOT / "svg_t2i/pre-trained/svg_autoencoder_P_stage3_1024.yaml"
)
DEFAULT_DINOV3_STATS = SVG_DIT_ROOT / "pre-trained/dinov3_s16p_layer_patchtoken_stats_dist.pt"

SYSTEM_PROMPTS: Dict[str, str] = {
    "align": "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> ",
    "base": "You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> ",
    "aesthetics": "You are an assistant designed to generate high-quality images with highest degree of aesthetics based on user prompts. <Prompt Start> ",
    "real": "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start> ",
    "4grid": "You are an assistant designed to generate four high-quality images with highest degree of aesthetics arranged in 2x2 grids based on user prompts. <Prompt Start> ",
    "empty": "",
    "enhance": "You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> ",
}

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _validate_path(path: Path, desc: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found at {path}")
    return path


@dataclass
class ResolutionSpec:
    """Parsed resolution specification."""

    name: str
    width: int
    height: int

    @property
    def latent_size(self) -> Tuple[int, int]:
        return self.width // 16, self.height // 16

    @property
    def tag(self) -> str:
        base = f"{self.width}x{self.height}"
        return self.name.replace(":", "_") + f"_{base}" if self.name else base


class SVGGenerator:
    """Thin wrapper around the SVG-DiT sampling stack."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("SVG generation currently requires a CUDA-capable GPU.")

        self.dtype = DTYPE_MAP[args.precision]
        self.autocast_enabled = self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)

        self.tokenizer, self.text_encoder = self._load_text_encoder(args)
        self.autoencoder_cfg = OmegaConf.load(args.autoencoder_config)
        self.autoencoder_cfg.model.params.is_train = False
        self.autoencoder, self.z_channels, self.num_layer = self._load_autoencoder()
        self.dinov3_sp_mean, self.dinov3_sp_std = self._load_dinov3_stats(args, self.z_channels, self.num_layer)
        self.train_args = torch.load(Path(args.ckpt) / "model_args.pth", map_location="cpu") #, weights_only=False)
        self.latent_channels = getattr(self.train_args, "in_channels", 384)
        self.model = self._load_dit_model(args)
        self.sampler = self._build_sampler(args)

    def _load_text_encoder(self, args: argparse.Namespace):
        use_auth_token = args.hf_token if args.hf_token else None
        tokenizer = AutoTokenizer.from_pretrained(
            args.text_encoder,
            use_auth_token=use_auth_token,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "right"
        if self.args.system_type == "enhance":
            text_encoder = AutoModelForCausalLM.from_pretrained(
            args.text_encoder,
            torch_dtype=self.dtype,
            use_auth_token=use_auth_token,
            trust_remote_code=True,
            device_map="cuda",
        ).eval()
        else:
            text_encoder = AutoModel.from_pretrained(
                args.text_encoder,
                torch_dtype=self.dtype,
                use_auth_token=use_auth_token,
                trust_remote_code=True,
                device_map="cuda",
            ).eval()
        return tokenizer, text_encoder

    def _load_autoencoder(self) -> Tuple[torch.nn.Module, int, int]:
        model = instantiate_from_config(self.autoencoder_cfg.model).cuda().eval()
        params = self.autoencoder_cfg.model.params
        if getattr(params, "ddconfig", None) is not None:
            z_channels = params.ddconfig.z_channels
        else:
            z_channels = params.hyconfig.z_channels
        num_layer = params.get("num_layer", -1)
        return model, z_channels, num_layer

    def _load_dinov3_stats(self, args: argparse.Namespace, z_channels: int, num_layer: int):
        stats = torch.load(args.dinov3_stats, map_location=self.device)
        dinov3_sp_mean = stats["mean"][num_layer][None, None, :].to(self.device)
        dinov3_sp_std = stats["std"][num_layer][None, None, :].to(self.device)
        if z_channels == 392:
            mean_extra = dinov3_sp_mean.mean(-1, keepdim=True).expand(-1, -1, 8).to(self.device)
            std_extra = dinov3_sp_std.mean(-1, keepdim=True).expand(-1, -1, 8).to(self.device)
            dinov3_sp_mean = torch.cat([dinov3_sp_mean, mean_extra], dim=-1)
            dinov3_sp_std = torch.cat([dinov3_sp_std, std_extra], dim=-1)
        return dinov3_sp_mean, dinov3_sp_std

    def _load_dit_model(self, args: argparse.Namespace):
        cap_feat_dim = self.text_encoder.config.hidden_size
        model_cls = models.__dict__[self.train_args.model]
        model = model_cls(in_channels=self.latent_channels, qk_norm=self.train_args.qk_norm, cap_feat_dim=cap_feat_dim)
        ckpt_path = self._resolve_checkpoint_file(Path(args.ckpt), args)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval().to("cuda", dtype=self.dtype)
        return model

    def _resolve_checkpoint_file(self, ckpt_dir: Path, args: argparse.Namespace) -> Path:
        candidates = []
        shard_suffix = "00-of-01"
        if args.ema:
            candidates.append(ckpt_dir / f"consolidated.ema.{shard_suffix}.pth")
            candidates.append(ckpt_dir / "ema" / f"consolidated.{shard_suffix}.pth")
        candidates.append(ckpt_dir / f"consolidated.{shard_suffix}.pth")
        candidates.append(ckpt_dir / "consolidated.pth")
        for cand in candidates:
            if cand.exists():
                return cand
        shards = sorted(ckpt_dir.glob("consolidated.*.pth"))
        if shards:
            return shards[0]
        pths = sorted(ckpt_dir.glob("*.pth"))
        if not pths:
            raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
        return pths[0]

    def _build_sampler(self, args: argparse.Namespace) -> Sampler:
        if args.solver == "dpm":
            transport = create_transport("Linear", "velocity")
        else:
            transport = create_transport(
                args.path_type,
                args.prediction,
                args.loss_weight,
                args.train_eps,
                args.sample_eps,
            )
        return Sampler(transport)

    def build_system_prompt(self, system_type: str) -> str:
        if system_type not in SYSTEM_PROMPTS:
            raise ValueError(f"Unsupported system_type: {system_type}")
        return SYSTEM_PROMPTS[system_type]

    @torch.inference_mode()
    def generate_image(
        self,
        full_prompt: str,
        negative_prompt: str,
        resolution: ResolutionSpec,
        sample_seed: int,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        width, height = resolution.width, resolution.height
        latent_w, latent_h = resolution.latent_size
        z = torch.randn((1, self.latent_channels, latent_h, latent_w), device="cuda", dtype=self.dtype)
        z = z.repeat(2, 1, 1, 1)

        prompts = [full_prompt, negative_prompt if negative_prompt else ""]
      
        cap_feats, cap_mask = encode_prompt(prompts, self.text_encoder, self.tokenizer, 0.0, is_train=False)
        cap_mask = cap_mask.to(cap_feats.device)
        model_kwargs = dict(cap_feats=cap_feats, cap_mask=cap_mask, cfg_scale=self.args.cfg_scale)

        start = time.perf_counter()
        if self.autocast_enabled:
            context = torch.autocast("cuda", dtype=self.dtype)
        else:
            context = torch.autocast("cuda", dtype=self.dtype, enabled=False)
        with context:
            if self.args.solver == "dpm":
                sample_fn = self.sampler.sample_dpm(self.model.forward_with_cfg, model_kwargs=model_kwargs)
                samples = sample_fn(
                    z,
                    steps=self.args.steps,
                    order=2,
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=self.args.time_shifting_factor,
                )
            else:
                sample_fn = self.sampler.sample_ode(
                    sampling_method=self.args.solver,
                    num_steps=self.args.steps,
                    atol=self.args.atol,
                    rtol=self.args.rtol,
                    reverse=self.args.reverse,
                    time_shifting_factor=self.args.time_shifting_factor,
                )
                samples = sample_fn(z, self.model.forward_with_cfg, **model_kwargs)[-1]
        torch.cuda.synchronize()
        end = time.perf_counter()

        samples = samples[:1]
        feats = samples.reshape(samples.shape[0], samples.shape[1], -1).permute(0, 2, 1)
        feats = feats * self.dinov3_sp_std.to(feats.device) + self.dinov3_sp_mean.to(feats.device)
        feats = feats.permute(0, 2, 1).reshape(samples.shape)
        recon = self.autoencoder.decode(feats)
        recon = (recon + 1.0) / 2.0
        recon.clamp_(0.0, 1.0)
        image_tensor = recon[0].detach().mul(255).permute(1, 2, 0).byte().cpu().numpy()
        image = Image.fromarray(image_tensor)

        return image, {
            "sampling_time": end - start,
            "solver": self.args.solver,
            "width": width,
            "height": height,
        }


def parse_resolution(spec: str) -> ResolutionSpec:
    if ":" in spec:
        name, dims = spec.split(":", 1)
    else:
        name, dims = spec, spec
    parts = dims.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution specification: {spec}")
    width, height = (int(parts[0]), int(parts[1]))
    if width % 16 or height % 16:
        raise ValueError("Resolution must be divisible by 16 in both dimensions")
    return ResolutionSpec(name=name, width=width, height=height)


def load_metadata(path: Path) -> List[Any]:
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, list):
            return data
        return [data]

    entries: List[Any] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                entries.append(line)
    return entries


def extract_prompt(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if not isinstance(entry, dict):
        raise ValueError("Metadata entry must be dict or string")
    for key in ("refined_prompt", "prompt", "gpt_4_caption", "caption", "text", "description", "query"):
        value = entry.get(key)
        if not value:
            continue
        if isinstance(value, list):
            return str(value[0])
        return str(value)
    raise KeyError("Unable to locate text prompt inside metadata entry")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images using the SVG-DiT model")
    parser.add_argument("metadata_file", type=str, help="JSON or JSONL file containing prompts")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to store generated images")
    parser.add_argument("--n-samples", dest="n_samples", type=int, default=4, help="Images to generate per prompt")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--time-shifting-factor", type=float, default=10.0, help="Flow time shifting for DPM solver")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--precision", type=str, choices=list(DTYPE_MAP.keys()), default="bf16")
    parser.add_argument("--solver", type=str, default="dpm", choices=["dpm", "midpoint", "euler", "heun"], help="Sampler backend")
    parser.add_argument("--system-type", type=str, default="base", choices=list(SYSTEM_PROMPTS.keys()))
    parser.add_argument("--skip-grid", action="store_true", help="Disable grid visualization")
    parser.add_argument("--grid-columns", type=int, default=2, help="Number of images per row in the grid")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt text")
    parser.add_argument("--ema", action="store_true", help="Use EMA weights if available")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for gated models")

    # Optional subset of metadata indices to process (for multi-GPU dispatch).
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index (inclusive) of prompts to process in the metadata file.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="End index (exclusive) of prompts to process in the metadata file. -1 means until the end.",
    )

    parser.add_argument(
        "--text-encoder",
        type=str,
        default=DEFAULT_TEXT_ENCODER,
        required=DEFAULT_TEXT_ENCODER is None,
        help="Path or HF repo id for the text encoder",
    )
    parser.add_argument(
        "--autoencoder-config",
        type=str,
        default=str(DEFAULT_AUTOENCODER_CONFIG),
        help="Path to SVG autoencoder config YAML",
    )
    parser.add_argument(
        "--dinov3-stats",
        type=str,
        default=str(DEFAULT_DINOV3_STATS),
        help="Path to DinoV3 statistics tensor",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        nargs="+",
        default=["512:512x512"],
        help="Resolution spec like '1024:720x1280'",
    )

    parse_transport_args(parser)
    parse_ode_args(parser)

    args = parser.parse_args()
    args.autoencoder_config = str(_validate_path(Path(args.autoencoder_config), "Autoencoder config"))
    args.dinov3_stats = str(_validate_path(Path(args.dinov3_stats), "DinoV3 stats"))
    args.ckpt = str(_validate_path(Path(args.ckpt), "Checkpoint directory"))
    return args


def main(args: argparse.Namespace) -> None:
    metadata_entries = load_metadata(Path(args.metadata_file))
    if not metadata_entries:
        raise RuntimeError("No metadata entries found")

    total = len(metadata_entries)
    start = max(args.start_index, 0)
    end = total if args.end_index < 0 else min(args.end_index, total)
    if start >= end:
        raise RuntimeError(f"Invalid index range: start_index={start}, end_index={end}, total={total}")

    generator = SVGGenerator(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    resolutions = [parse_resolution(spec) for spec in args.resolution]

    for index in tqdm(range(start, end), desc="Prompts"):
        metadata = metadata_entries[index]
        prompt_text = extract_prompt(metadata)
        system_prompt = generator.build_system_prompt(args.system_type)
        
        if args.system_type == "enhance":
            prompt_text = generate_enhanced_prompt(
                prompt_text, generator.text_encoder, generator.tokenizer, generator.device
            )
            
    
        full_prompt = f"{system_prompt}{prompt_text}"
        # print(f'full_prompt = {full_prompt}')
        # import pdb; pdb.set_trace()
        neg_prompt = args.negative_prompt.strip()
        full_neg_prompt = f"{system_prompt}{neg_prompt}" if neg_prompt else ""

        for res in resolutions:
            outpath = outdir / f"{index:05d}" / res.tag
            samples_dir = outpath / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)

            tensors_for_grid: List[torch.Tensor] = []
            sample_records: List[Dict[str, Any]] = []

            for sample_idx in range(args.n_samples):
                sample_seed = args.seed + index * args.n_samples + sample_idx
                seed_everything(sample_seed)
                image, info = generator.generate_image(full_prompt, full_neg_prompt, res, sample_seed)
                filename = f"{sample_idx:05d}.png"
                image.save(samples_dir / filename)
                tensors_for_grid.append(to_tensor(image))
                sample_records.append(
                    {
                        "file_name": filename,
                        "seed": sample_seed,
                        "sampling_time": info["sampling_time"],
                        "solver": info["solver"],
                        "steps": args.steps,
                        "cfg_scale": args.cfg_scale,
                        "resolution": f"{res.width}x{res.height}",
                    }
                )

            if tensors_for_grid and not args.skip_grid:
                grid = make_grid(
                    torch.stack(tensors_for_grid, dim=0),
                    nrow=min(args.grid_columns, len(tensors_for_grid)),
                )
                grid = (grid.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy())
                Image.fromarray(grid).save(outpath / "grid.png")
                
                # Save to grid_output folder if filename exists in metadata
                if isinstance(metadata, dict) and "filename" in metadata:
                    filename = metadata["filename"]
                    # Extract base name without extension and use .png
                    base_name = Path(filename).stem
                    grid_output_dir = outdir / "grid_output"
                    grid_output_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(grid).save(grid_output_dir / f"{base_name}.png")

            manifest = {
                "prompt_index": index,
                "prompt": prompt_text,
                "full_prompt": full_prompt,
                "negative_prompt": neg_prompt,
                "system_type": args.system_type,
                "resolution": f"{res.width}x{res.height}",
                "samples": sample_records,
                "metadata": metadata,
            }
            outpath.mkdir(parents=True, exist_ok=True)
            with open(outpath / "metadata.jsonl", "w", encoding="utf-8") as fp:
                fp.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main(parse_args())
