#!/usr/bin/env sh
set -e

# ========== Setup ==========
res="256:256x256"
res="512:512x512"
res="1024:1024x1024"

seed=0
time_shifting_factor=10
cfg_scale=4
steps=50
solver=dpm # midpoint, euler, dpm
system_type=base

# caption
cap_dir=configs/sample_caption.jsonl
cap_dir=configs/example.jsonl
autoencoder_path=pre-trained/autoencoder/svg_autoencoder_P_stage3_1024.yaml
ckpt_dir=pre-trained/dit-stage4-T274M/
exp_name=$(basename $ckpt_dir)

out_dir=samples/infer_${exp_name}_${res}_seed${seed}_steps${steps}_${system_type}_${solver}_cfg${cfg_scale}_shift${time_shifting_factor}_pretrained_ema

echo "========== Running ckpt: $ckpt =========="
echo "Model Dir: $ckpt_dir"
echo "Output Dir: $out_dir"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python -u sample_svg_t2i.py --ckpt ${ckpt_dir} \
    --image_save_path ${out_dir} \
    --solver ${solver} --num_sampling_steps ${steps} \
    --caption_path ${cap_dir} \
    --seed ${seed} \
    --resolution ${res} \
    --time_shifting_factor ${time_shifting_factor} \
    --cfg_scale ${cfg_scale} \
    --system_type ${system_type} \
    --autoencoder_path ${autoencoder_path} \
    --batch_size 1 \
    --rank 0 \
    --ema
