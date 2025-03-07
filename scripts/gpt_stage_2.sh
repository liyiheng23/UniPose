#!/bin/bash
# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!
# Uncomment and set the following variables correspondingly to run this script:
MODEL_VERSION=llava-v1.6-mistral-7b
STAGE=2-unfrozen-vit
EPOCH=2
TRAIN_LLM=False
TRAIN_MM_PROJECTOR=True
LOAD_FROM=./checkpoints/stage-1-epoch-10/checkpoint-34110
deepspeed \
    --include=localhost:0,1,2,3 \
    --master_port=29501 \
    scripts/gpt_train.py \
    --deepspeed ./scripts/zero2.json \
    --config configs/posegpt_pretrain_stage_2.py \
    --lora_enable True \
    --vision_tower cache/clip-vit-large-patch14-336 \
    --model_name_or_path ./cache/$MODEL_VERSION \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --output_dir ./checkpoints/stage-$STAGE-epoch-$EPOCH \
    --load_from $LOAD_FROM \
    --train_llm $TRAIN_LLM \
    --train_mm_projector $TRAIN_MM_PROJECTOR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 8 \
    --mm_projector_type mlp2x_gelu \
    --mm_patch_merge_type spatial_unpad \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard
