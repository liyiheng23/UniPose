name='pose_vqvae'
accelerator='gpu' # Devices optioncal: “cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”
num_nodes=1 # Number of GPU nodes for distributed training
device=[0] # Index of gpus eg. [0] or [0,1,2,3]
eval_interval=1
max_epochs=240

model=dict(
    target='posegpt.models.pose_vqvae.PoseVQVAE', 
    params=dict(
        encoder=dict(
            target='posegpt.models.pose_vqvae.TransformerPoseEncoderV1',
            params=dict(
                input_dim=6, 
                hidden_dim=512, 
                output_dim=256, 
            )
        ), 
        decoder=dict(
            target='posegpt.models.pose_vqvae.TransformerPoseDecoderV1', 
            params=dict(
                input_dim=256, 
                hidden_dim=512, 
                output_dim=6,
                num_joints=22)), 

        quantizer=dict(
            target='posegpt.models.components.quantize_cnn.QuantizeEMAReset', 
            params=dict(
                nb_code=2048, 
                code_dim=256, 
                mu=0.99
            ), 
        ), 
    ), 

    loss_config=dict(
        target='posegpt.models.losses.PoseVQLoss', 
        params=dict(
            lambda_pose=20.0, 
            lambda_joints=100.0, 
            lambda_vertices=100.0, 
            lambda_commit=1.0, 
        )
    ),

    metric_config=dict(
        target='posegpt.models.metrics.PoseReconstruction_SMPL_Metric', 
        params=dict(), 
    ), 

    optimizer_config=dict(
        target='torch.optim.AdamW', 
        params=dict(
            lr=2e-4, 
            betas=(0.9, 0.99), 
            weight_decay=1e-5)), 

    lr_config=dict(
        target='torch.optim.lr_scheduler.MultiStepLR', 
        params=dict(
            milestones=[int(max_epochs * 8 / 12), int(max_epochs * 11 / 12)], 
            gamma=0.1)), 

    checkpoint_config=dict(
        monitor_variable_name='MPJPE', 
        monitor_variable_mode='min', 
    )
)

data=dict(
    target='posegpt.datasets.pose_vq.MixedPoseVQDataset', 
    train=dict(
        samples_per_gpu=256,
        workers_per_gpu=16,
        split='train',
        dataset_root='processed_dataset/pose_vq_dataset', 
        dataset_list = ['CMU', 'KIT', 'BMLrub', 'DanceDB', 'BMLmovi', 'EyesJapan', 'BMLhandball', 'TotalCapture', 'EKUT', 'ACCAD', 'TCDHands', 'MPI-Limits', 'MOYO'], 
        dataset_partition = [0.12, 0.11, 0.11, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.16, 0.16], 
        smpl_path='processed_dataset/smpl_models/smplh/SMPLH_NEUTRAL.npz', 
    ), 
    val=dict(
        samples_per_gpu=256,
        workers_per_gpu=16,
        split='val',
        dataset_root='processed_dataset/pose_vq_dataset', 
        dataset_list = ['MOYO'], 
        smpl_path='processed_dataset/smpl_models/smplh/SMPLH_NEUTRAL.npz', 
    ), 
    test=dict(
        samples_per_gpu=256,
        workers_per_gpu=16,
        split='test',
        dataset_root='processed_dataset/pose_vq_dataset', 
        dataset_list = ['SSM', 'Transitions'], 
        smpl_path='processed_dataset/smpl_models/smplh/SMPLH_NEUTRAL.npz', 
    ),
)
