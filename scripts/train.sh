#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

# text pose retrieval
# python -m scripts.train --config configs/retrieval_text_pose.py \
#     --load-from processed_dataset/saved_checkpoints/retrieval_text_pose_vqpose_pretrain/best_mRecall.ckpt

# text poseAB retrieval pretrain
# python -m scripts.train --config configs/retrieval_text_poseAB.py

# text poseAB retrieval finetune
# python -m scripts.train --config configs/retrieval_text_poseAB.py \
#     --load-from processed_dataset/saved_checkpoints/retrieval_text_poseAB_vqpose_pretrain/best_mRecall.ckpt

# stage 1: pose vae
python -m scripts.train --config configs/pose_vqvae.py
