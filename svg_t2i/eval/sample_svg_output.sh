#!/usr/bin/env bash
set -e

#############################################
#                 通用配置
#############################################

# 输入 prompt 文件
PROMPT_FILE="prompts/dpg.jsonl"

PROMPT_NAME=$(basename "$PROMPT_FILE" .jsonl)

RES="1024:1024x1024"
SYSTEM_TYPE="base"
GPU_LIST="0,1,2,3,4,5,6,7"

OUTDIR="svg_output_${SYSTEM_TYPE}_${PROMPT_NAME}"
RESULT_JSON="result/results_${SYSTEM_TYPE}_${PROMPT_NAME}_${RES//:/_}.jsonl"

# 预训练模型路径
CKPT="pretrained_models/model.ckpt"

#############################################
#                 执行流程
#############################################

echo "Running SVG generation..."
echo "  Prompt file : $PROMPT_FILE"
echo "  Outdir      : $OUTDIR"
echo "  Checkpoint  : $CKPT"

conda activate geneval

python generation/svg_generate_dist.py \
    "$PROMPT_FILE" \
    --ckpt "$CKPT" \
    --outdir "$OUTDIR" \
    --gpus "$GPU_LIST" \
    --resolution "$RES" \
    --system-type "$SYSTEM_TYPE"