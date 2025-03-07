#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
# text pose match
# python -m scripts.test \
#     --config work_dirs/retrieval_text_pose_vqpose_finetune/retrieval_text_pose.py \
#     --checkpoints work_dirs/retrieval_text_pose_vqpose_finetune/epoch=109.ckpt

# text poseAB match
python -m scripts.test \
    --config work_dirs/pose_vqvae_transformer/pose_vqvae_transformer.py \
    --checkpoints work_dirs/pose_vqvae_transformer/best_MPJPE.ckpt

# stage 1: pose vae
# python -m scripts.test \
#     --config work_dirs/pose_vqvae/pose_vqvae.py \
#     --checkpoints work_dirs/pose_vqvae/best_MPJPE.ckpt

