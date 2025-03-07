#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_PATH=_checkpoints/stage-ablation-clip-frozen-fix-hmr-bug-epoch-2/checkpoint-6250
# accelerate launch \
#     --main_process_port 29501 \
#     scripts/gpt_eval.py \
#     --config configs/posegpt_eval_finetune_posescript.py \
#     --model-base cache/llava-v1.6-mistral-7b \
#     --model-path $MODEL_PATH \
#     --bf16

accelerate launch \
    --main_process_port 29502 \
    scripts/gpt_eval_clip.py \
    --config configs/posegpt_eval_finetune.py \
    --model-base cache/llava-v1.6-mistral-7b \
    --model-path $MODEL_PATH \
    --bf16
