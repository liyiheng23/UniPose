#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
MODEL_PATH=_checkpoints_backup/stage-finetune-full-mask-instructions-fix-hmr-bug-epoch-4/checkpoint-4840

# accelerate launch \
#     --main_process_port 29501 \
#     scripts/gpt_eval.py \
#     --config configs/posegpt_eval_finetune.py \
#     --model-base cache/llava-v1.6-mistral-7b \
#     --model-path $MODEL_PATH \
#     --bf16

# accelerate launch \
#     --main_process_port 29500 \
#     scripts/gpt_eval.py \
#     --config configs/posegpt_pretrain_stage_1_full_mask_debug.py \
#     --model-base cache/llava-v1.6-mistral-7b \
#     --model-path $MODEL_PATH \
#     --bf16

# full mask + finetune
accelerate launch \
    --main_process_port 29500 \
    scripts/gpt_eval_full_mask.py \
    --config configs/posegpt_eval_finetune.py \
    --model-base cache/llava-v1.6-mistral-7b \
    --model-path $MODEL_PATH \
    --bf16

# mlp
# accelerate launch \
#     --main_process_port 29500 \
#     scripts/gpt_eval_mlp.py \
#     --config configs/posegpt_eval_finetune.py \
#     --model-base cache/llava-v1.6-mistral-7b \
#     --model-path $MODEL_PATH \
#     --bf16