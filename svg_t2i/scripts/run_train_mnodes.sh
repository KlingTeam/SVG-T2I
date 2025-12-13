#!/usr/bin/env bash
set -e

# ========== Setup ==========
node_rank=$1
NNODES=4
NPROC_PER_NODE=8
MASTER_ADDR="0.0.0.0"
MASTER_PORT=12345
size=256
train_data_path='./configs/example.yaml'
model=NextDiT_2B_GQA_patch1_Adaln_Refiner
batch_size=16
global_bsz=$((batch_size * NPROC_PER_NODE * NNODES))

if [ "$node_rank" -eq 0 ]; then
    mkdir -p results/"$exp_name"
    log_cmd="2>&1 | tee -a results/${exp_name}/output_rank0.log"
else
    log_cmd=""
fi

precision=bf16
data_parallel=sdp
lr=2e-4
num_workers=1
snr_type=lognorm
mu=0.0
exp_name=stage1_${model}_bs${batch_size}_gbs${global_bsz}_${precision}_${NNODES}x${NPROC_PER_NODE}_lr${lr}_${data_parallel}_nw${num_workers}_snr${snr_type}

echo "🚀 Starting training on rank $node_rank (Master: ${MASTER_ADDR}:${MASTER_PORT})"

eval torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=$node_rank \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_svg_t2i.py \
    --global_bsz $global_bsz \
    --micro_bsz $batch_size \
    --resol ${size} \
    --model ${model} \
    --lr ${lr} --grad_clip 2.0 \
    --precision ${precision} --qk_norm \
    --data_path ${train_data_path} \
    --results_dir results/${exp_name} \
    --num_workers ${num_workers} \
    --no_auto_resume \
    --max_steps 3000000 \
    --ckpt_every 2000 --log_every 10 \
    --data_parallel ${data_parallel} \
    --snr_type ${snr_type} \
    --mu  ${mu} \
    --checkpointing \
    --max_cap_len 512 \
    --use_long_cap \
    --init_from null \
    --gemma2b google/gemma-2-2b \
    --autoencoder_path pre-trained/autoencoder/svg_autoencoder_R_stage3_1024.yaml \
    $log_cmd
