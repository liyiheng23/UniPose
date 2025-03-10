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
pose_vqvae_ckp_path='cache/pose_vqvae/best_MPJPE.ckpt'

# ====================================================
# hmr vit 
# ====================================================
hmr_vit_ckp_path = 'cache/tokenhmr_model.ckpt'
