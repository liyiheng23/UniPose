# ====================================================
# PoseVQVAE
# ====================================================
pose_vqvae_config=dict(
    target='posegpt.models.pose_vqvae.PoseVQVAE', 
    params=dict(
        encoder=dict(
            target='posegpt.models.pose_vqvae.TokenHMREncoder',
            params=dict(
                input_dim=6, 
                hidden_dim=512, 
                output_dim=256, 
                num_joints=22, 
                upsample_step=2, 
                downsample_step=1)), 
        decoder=dict(
            target='posegpt.models.pose_vqvae.TokenHMRDecoder', 
            params=dict(
                input_dim=256, 
                hidden_dim=512, 
                output_dim=6,
                num_joints=22, 
                upsample_step=2, 
                downsample_step=1)), 

        quantizer=dict(
            target='posegpt.models.components.quantize_cnn.QuantizeEMAReset', 
            params=dict(
                nb_code=2048, 
                code_dim=256, 
                mu=0.99)), 
    )
)
pose_vqvae_ckp_path='cache/saved_checkpoints/pose_vqvae_noise_root_orient/best_MPJPE.ckpt'

# ====================================================
# hmr vit 
# ====================================================
hmr_vit_ckp_path = 'cache/tokenhmr_model.ckpt'

# ====================================================
# Metric
# ====================================================

pose_text_encoder_config=dict(
    target='posegpt.models.retrieval.TextPoseRetrieval', 
    params=dict(
        pose_encoder=dict(
            target='posegpt.models.retrieval.VQPoseEncoder',
            params=dict(
                input_dim=6, # 6d rot
                hidden_dim=512, 
                output_dim=512, 
                upsample_step=1, 
                downsample_step=1,
                num_joints=22)
        ), 

        text_encoder=dict(
            target='posegpt.models.retrieval.TextEncoderBiGRUCo', 
            params=dict(
                hidden_dim=512, 
                output_dim=512)),  
    ), 
)

poseAB_text_encoder_config=dict(
    target='posegpt.models.retrieval.TextPoseABRetrieval', 
    params=dict(
        pose_encoder=dict(
            target='posegpt.models.retrieval.VQPoseABEncoder',
            params=dict(
                input_dim=6, # 6d rot
                hidden_dim=512, 
                output_dim=512, 
                upsample_step=1, 
                downsample_step=1,
                num_joints=22)
        ), 

        text_encoder=dict(
            target='posegpt.models.retrieval.TextEncoderBiGRUCo', 
            params=dict(
                hidden_dim=512, 
                output_dim=512)),  
    ), 
)

pose_text_encoder_ckp_path = 'cache/saved_checkpoints/retrieval_text_pose_vqpose_finetune_trainval/best_mRecall.ckpt'
poseAB_text_encoder_ckp_path = 'cache/saved_checkpoints/retrieval_text_poseAB_vqpose_finetune_trainval/best_mRecall.ckpt'

metrics = dict(
    text2pose=dict(
        target='posegpt.models.metrics.Text2PoseMetric', 
        params=dict(
            pose_text_encoder_config=pose_text_encoder_config, 
            pose_text_encoder_ckp_path=pose_text_encoder_ckp_path)
    ), 
    pose2text=dict(
        target='posegpt.models.metrics.Pose2TextMetric', 
        params=dict(
            pose_text_encoder_config=pose_text_encoder_config, 
            pose_text_encoder_ckp_path=pose_text_encoder_ckp_path)
    ), 
    image2text=dict(
        target='posegpt.models.metrics.Pose2TextMetric', 
        params=dict(
            pose_text_encoder_config=pose_text_encoder_config, 
            pose_text_encoder_ckp_path=pose_text_encoder_ckp_path)
    ), 
    image2text_reasoning=dict(
        target='posegpt.models.metrics.Pose2TextMetric', 
        params=dict(
            pose_text_encoder_config=pose_text_encoder_config, 
            pose_text_encoder_ckp_path=pose_text_encoder_ckp_path)
    ), 
    image_difference=dict(
        target='posegpt.models.metrics.PoseDifferenceMetric', 
        params=dict(
            pose_text_encoder_config=poseAB_text_encoder_config, 
            pose_text_encoder_ckp_path=poseAB_text_encoder_ckp_path
        ), 
    ), 
    image2pose=dict(
        target='posegpt.models.metrics.PoseReconstructionMetric', 
        params=dict(), 
    ), 
    image2pose_reasoning=dict(
        target='posegpt.models.metrics.PoseReconstructionMetric', 
        params=dict(), 
    ), 
    pose_edit=dict(
        target='posegpt.models.metrics.PoseEditMetric', 
        params=dict(
            pose_text_encoder_config=pose_text_encoder_config, 
            pose_text_encoder_ckp_path=pose_text_encoder_ckp_path
        ), 
    ), 
    pose_difference=dict(
        target='posegpt.models.metrics.PoseDifferenceMetric', 
        params=dict(
            pose_text_encoder_config=poseAB_text_encoder_config, 
            pose_text_encoder_ckp_path=poseAB_text_encoder_ckp_path
        ), 
    ), 
)