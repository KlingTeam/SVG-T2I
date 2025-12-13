import argparse
import json
import math
import os
import random
import socket
import time

from diffusers.models import AutoencoderKL
import numpy as np
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

import models
from transport import Sampler, create_transport
from util import instantiate_from_config, SmoothedValue
from omegaconf import OmegaConf
from typing import TypeVar 
T = TypeVar('T')

def generate_enhanced_prompt(
    prompt: str,
    text_encoder,
    tokenizer,
    device: str,
    max_new_tokens: int = 196,
) -> str:
    """
    Use the Gemma LLM to generate an enhanced prompt.
    Make the description more detailed, while strictly preserving subjects and constraints.
    """
    # Use a strongly constrained instruction to guarantee subject and constraint preservation
    input_text = (
        "You are an assistant that rewrites image generation prompts.\n"
        "Given a user prompt, rewrite it into a more detailed and concrete visual description.\n"
        "You MUST strictly preserve:\n"
        "- the main subject(s) and their roles;\n"
        "- the number of each entity (do not change counts like 'one', 'two', 'three');\n"
        "- explicit attributes (e.g. colors, sizes, poses) that appear in the user prompt;\n"
        "- explicit spatial relationships (left/right, front/behind, on top of, in the background, etc.);\n"
        "- explicit style or format constraints (e.g. '3D render', 'line art', 'pixel art', 'photograph').\n"
        "You may only add a few plausible visual details (textures, lighting, background context) "
        "that do NOT contradict any part of the original prompt.\n"
        "Avoid repeating the same subject or action multiple times; keep the description concise.\n"
        "Do not remove or alter any constraints given in the user prompt.\n"
        "Do NOT mention the words 'User Prompt' or 'Enhanced Prompt' in your answer.\n"
        "Output only the enhanced description as one or two sentences.\n\n"
        f"User prompt: {prompt}\n"
        "Enhanced description:"
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = text_encoder.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Only take newly generated tokens
    generated_tokens = outputs[0, input_length:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # --- Post-processing: ensure a concise, non-repetitive enhanced description ---
    text = raw_text

    # If multiple lines are generated, keep only the first non-empty one
    if "\n" in text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
        if len(lines) > 0:
            text = lines[0]

    # Remove possible prefix labels such as "Enhanced description:", "Enhanced:" etc.
    if ":" in text[:40]:
        head, tail = text.split(":", 1)
        # Cut only if the prefix looks like a label
        if any(key in head.lower() for key in ["enhanced", "description", "prompt"]):
            text = tail.strip()

    # Roughly split by commas, deduplicate clauses, and limit total length
    clauses = [c.strip() for c in text.split(",") if c.strip()]
    if clauses:
        seen = set()
        dedup_clauses = []
        for c in clauses:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup_clauses.append(c)
            # Keep at most the first 4 unique clauses to avoid overly long descriptions
            if len(dedup_clauses) >= 4:
                break
        text = ", ".join(dedup_clauses)

    # Character-level truncation to avoid extremely long outputs
    max_chars = 256
    if len(text) > max_chars:
        # Try to truncate at a space to avoid cutting words in half
        truncated = text[:max_chars]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        text = truncated

    enhanced = text.strip()

    # Fallback: if empty, return the original prompt
    if enhanced == "":
        enhanced = prompt

    return enhanced


def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument(
        "--path-type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "VP"],
        help="the type of path for transport: 'Linear', 'GVP' (Geodesic Vector Pursuit), or 'VP' (Vector Pursuit).",
    )
    group.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
        help="the prediction model for the transport dynamics.",
    )
    group.add_argument(
        "--loss-weight",
        type=none_or_str,
        default=None,
        choices=[None, "velocity", "likelihood"],
        help="the weighting of different components in the loss function, can be 'velocity' for dynamic modeling, 'likelihood' for statistical consistency, or None for no weighting.",
    )
    group.add_argument("--sample-eps", type=float, help="sampling in the transport model.")
    group.add_argument("--train-eps", type=float, help="training to stabilize the learning process.")


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the ODE solver.",
    )
    group.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the ODE solver.",
    )
    group.add_argument("--reverse", action="store_true", help="run the ODE solver in reverse.")
    group.add_argument(
        "--likelihood",
        action="store_true",
        help="Enable calculation of likelihood during the ODE solving process.",
    )



def none_or_str(value):
    if value == "None":
        return None
    return value


def main(args, rank, master_port):
    # Setup PyTorch:
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = rank % torch.cuda.device_count()
    device_str = f"cuda:{device}"
    
    if dist.get_rank() == 0:
        print(f"Creating text encoder: {args.text_encoder}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    gemma_2_ckpt = "/ytech_m2v3_hdd/yuanziyang/sml/opensorce_ckpt/gemma-2-2b"

    tokenizer = AutoTokenizer.from_pretrained(gemma_2_ckpt)
    tokenizer.padding_side = "right"

    if args.system_type == "enhance":
        text_encoder = AutoModelForCausalLM.from_pretrained(
            gemma_2_ckpt, torch_dtype=dtype, device_map="cuda"
        ).eval()
    else:
        text_encoder = AutoModel.from_pretrained(
            gemma_2_ckpt, torch_dtype=dtype, device_map="cuda"
        ).eval()

    
    cap_feat_dim = text_encoder.config.hidden_size

    encoder_config = OmegaConf.load(args.autoencoder_path)
    encoder_config.model.params.is_train = False
    dinov3 = instantiate_from_config(encoder_config.model).cuda().eval()
    z_channels = encoder_config.model.params.ddconfig.z_channels if encoder_config.model.params.ddconfig is not None else encoder_config.model.params.hyconfig.z_channels
    num_layer = encoder_config.model.params.get("num_layer", -1)
    dinov3_stats = torch.load("pre-trained/dinov3_s16p_layer_patchtoken_stats_dist.pt")
    dinov3_sp_mean = dinov3_stats["mean"][num_layer][None, None, :].to(device)
    dinov3_sp_std = dinov3_stats["std"][num_layer][None, None, :].to(device)
    if z_channels == 392:
        mean_extra = dinov3_sp_mean.mean(-1, keepdim=True).expand(-1, -1, 8).to(device)
        std_extra = dinov3_sp_std.mean(-1, keepdim=True).expand(-1, -1, 8).to(device)
        dinov3_sp_mean = torch.cat([dinov3_sp_mean, mean_extra], dim=-1).to(device)
        dinov3_sp_std = torch.cat([dinov3_sp_std, std_extra], dim=-1).to(device)


    if dist.get_rank() == 0:
        print(f"Creating DiT: {args.model}")
    model = models.__dict__[args.model](
        in_channels=384,
        qk_norm=True,
        cap_feat_dim=cap_feat_dim,
    )
    model.eval().to("cuda", dtype=dtype)
    
    if args.debug == False:
        if len(args.direct_ckpt) > 0:
            ckpt = torch.load(
                os.path.join(
                    args.direct_ckpt
                )
            )["model"]
            model.load_state_dict(ckpt, strict=True)
        else:
            if args.ema:
                print("Loading ema model.")
            ckpt = torch.load(
                os.path.join(
                    args.ckpt,
                    f"consolidated.{rank:02d}-of-{args.num_gpus:02d}.pth",
                )
            )
            model.load_state_dict(ckpt, strict=True)
    
    # begin sampler``
    if args.solver == "dpm":
        transport = create_transport(
            "Linear",
            "velocity",
        )
        sampler = Sampler(transport)
    else:
        transport = create_transport(
            args.path_type,
            args.prediction,
            args.loss_weight,
            args.train_eps,
            args.sample_eps,
        )
        sampler = Sampler(transport)

    sample_folder_dir = args.image_save_path

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_folder_dir, "images"), exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    info_path = os.path.join(args.image_save_path, "data.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.loads(f.read())
        collected_id = []
    else:
        info = []
        collected_id = []

    if args.caption_path.endswith("json"):
        with open(args.caption_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    elif args.caption_path.endswith("jsonl"):
        data = []
        with open(args.caption_path, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line))
    else:
        data = []
        with open(args.caption_path, "r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if text:
                    data.append(line.strip())
    neg_cap = ""
    total = len(info)
    resolution = args.resolution

    with torch.autocast("cuda", dtype):
        for res in resolution:
            for idx, item in tqdm(enumerate(data)):

                # if int(args.seed) != 0:
                    # torch.random.manual_seed(int(args.seed))

                sample_id = f'{idx}_{res.split(":")[-1]}'
                if isinstance(item, str):
                    cap = item
                else:
                    if "refined_prompt" in item:
                        cap = item["refined_prompt"]
                        print("refined_prompt")
                    elif "prompt" in item:
                        cap = item["prompt"]
                        print("prompt")
                    else:
                        cap = item["gpt_4_caption"]

                if args.system_type == "align":
                    system_prompt = "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> "  # noqa
                elif args.system_type == "base":
                    system_prompt = "You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> "  # noqa
                    # system_prompt = """Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
                    # - If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
                    # - If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.
                    # Here are examples of how to transform or refine prompts:
                    # - User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
                    # - User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.
                    # Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
                    # User Prompt:"""
                elif args.system_type == "aesthetics":
                    system_prompt = "You are an assistant designed to generate high-quality images with highest degree of aesthetics based on user prompts. <Prompt Start> "  # noqa
                elif args.system_type == "real":
                    system_prompt = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start> "  # noqa
                elif args.system_type == "4grid":
                    system_prompt = "You are an assistant designed to generate four high-quality images with highest degree of aesthetics arranged in 2x2 grids based on user prompts. <Prompt Start> "  # noqa
                elif args.system_type == "empty":
                    system_prompt = ""
                elif args.system_type == "enhance":
                    system_prompt = "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> " 
                else:
                    raise ValueError

                # 先根据原始 cap 通过 Gemma 生成增强版 prompt
                if args.system_type == "enhance":
                    enhanced_cap = generate_enhanced_prompt(
                        cap, text_encoder, tokenizer, device_str
                    )
                    cap = enhanced_cap
                    print(f'enhanced_cap = {cap}')

                cap = system_prompt + cap
                if neg_cap != "":
                    neg_cap = system_prompt + neg_cap

                caps_list = [cap]

                res_cat, resolution = res.split(":")
                res_cat = int(res_cat)
                
                n = len(caps_list)
                w, h = resolution.split("x")
                w, h = int(w), int(h)
                latent_w, latent_h = w // 16, h // 16
                z = torch.randn([1, 384, latent_w, latent_h], device="cuda").to(dtype)
                z = z.repeat(n * 2, 1, 1, 1)

                with torch.no_grad():
                    cap_feats, cap_mask = encode_prompt([cap] + [neg_cap], text_encoder, tokenizer, 0.0, is_train=False)

                # import ipdb; ipdb.set_trace()
                print(f"cap_feats shape: {cap_feats.shape}, cap_mask shape: {cap_mask.shape}")
                cap_mask = cap_mask.to(cap_feats.device)
                model_kwargs = dict(
                    cap_feats=cap_feats,
                    cap_mask=cap_mask,
                    cfg_scale=args.cfg_scale,
                )

                start_time = time.perf_counter()
                if args.solver == "dpm":
                    sample_fn = sampler.sample_dpm(
                        model.forward_with_cfg,
                        model_kwargs=model_kwargs,
                    )
                    samples = sample_fn(z, steps=args.num_sampling_steps, order=2, skip_type="time_uniform_flow", method="multistep", flow_shift=args.time_shifting_factor)
                else:
                    sample_fn = sampler.sample_ode(
                        sampling_method=args.solver,
                        num_steps=args.num_sampling_steps,
                        atol=args.atol,
                        rtol=args.rtol,
                        reverse=args.reverse,
                        time_shifting_factor=args.t_shift
                    )
                    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
                end_time = time.perf_counter()
                
                # import ipdb; ipdb.set_trace()
                samples = samples[:1]
                samples_dino_feature = samples  # B, D, H, W
                B, D, H, W = samples_dino_feature.shape
                samples_dino_feature = samples_dino_feature.reshape(B, D, H * W).permute(0, 2, 1)
                # un-normalize
                samples_dino_feature = samples_dino_feature * dinov3_sp_std.to(device) + dinov3_sp_mean.to(device)
                samples_dino_feature = samples_dino_feature.permute(0, 2, 1).reshape(B, D, H, W)
                # import ipdb; ipdb.set_trace()

                samples = dinov3.decode(samples_dino_feature)
                samples = (samples + 1.0) / 2.0
                
                samples.clamp_(0.0, 1.0)
                print("sample times:", end_time-start_time)

                # Save samples to disk as individual .png files
                for i, (sample, cap) in enumerate(zip(samples, caps_list)):
                    img = to_pil_image(sample.float())
                    save_path = f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}.png"
                    img.save(save_path)
                    info.append(
                        {
                            "caption": cap,
                            "image_url": f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}.png",
                            "resolution": f"res: {resolution}\ntime_shift: {args.time_shifting_factor}",
                            "solver": args.solver,
                            "num_sampling_steps": args.num_sampling_steps,
                        }
                    )

                with open(info_path, "w") as f:
                    f.write(json.dumps(info))

                total += len(samples)
                dist.barrier()
    # end_time = time.time()
    print("sample times:", end_time-start_time)
    dist.barrier()
    dist.barrier()
    dist.destroy_process_group()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--direct_ckpt", type=str, default="")
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--t_shift", type=int, default=6)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.set_defaults(ema=False)
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="samples",
        help="If specified, overrides the default image save path "
        "(sample{_ema}.png in the model checkpoint directory).",
    )
    parser.add_argument(
        "--time_shifting_factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="prompts.txt",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="",
        nargs="+",
    )
    parser.add_argument(
        "--autoencoder_path",
        type=str,
        default="",
    )
    parser.add_argument("--proportional_attn", type=bool, default=True)
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="None",
        choices=["Time-aware", "None"],
    )
    parser.add_argument(
        "--system_type",
        type=str,
        default="real",
        # choices=["Time-aware", "None"],
    )
    parser.add_argument(
        "--scaling_watershed",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse", "sdxl", "flux"], default="flux"
    )
    parser.add_argument(
        "--text_encoder", type=str, nargs='+', default=['gemma'], help="List of text encoders to use (e.g., t5, clip, gemma)"
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Max length for text encoder."
    )
    parser.add_argument(
        "--use_parallel_attn",
        type=bool,
        default=False,
        help="Use parallel attention in the model.",
    )
    parser.add_argument(
        "--use_flash_attn",
        type=bool,
        default=True,
        help="Use Flash Attention in the model.",
    )
    parser.add_argument("--do_shift", default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hf_token", type=str, default=None, help="huggingface read token for accessing gated repo.")
    parser.add_argument("--model", type=str, default="NextDiT_2B_GQA_patch1_Adaln_Refiner")

    parse_transport_args(parser)
    parse_ode_args(parser)

    args = parser.parse_known_args()[0]

    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)
