name='text_pose_retrieval'
accelerator='gpu' # Devices optioncal: “cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”
num_nodes=1 # Number of GPU nodes for distributed training
device=[0] # Index of gpus eg. [0] or [0,1,2,3]
eval_interval=5
max_epochs=120

model=dict(
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
                num_joints=22, 
            )
        ), 

        text_encoder=dict(
            target='posegpt.models.retrieval.TextEncoderBiGRUCo', 
            params=dict(
                hidden_dim=512, 
                output_dim=512, 
            ), 
        ),  
    ), 

    loss_config=dict(
        target='posegpt.models.losses.TextPoseMatchLoss', 
        params=dict()),

    metric_config=dict(
        target='posegpt.models.metrics.RetrievalMetric', 
        params=dict()), 

    optimizer_config=dict(
        target='torch.optim.Adam', 
        params=dict(
            lr=2e-4)), 

    lr_config=dict(
        target='torch.optim.lr_scheduler.MultiStepLR', 
        params=dict(
            milestones=[int(max_epochs * 8 / 12), int(max_epochs * 11 / 12)], 
            gamma=0.1)), 

    checkpoint_config=dict(
        monitor_variable_name='mRecall', 
        monitor_variable_mode='max', 
    )
)

data=dict(
    target='posegpt.datasets.retrieval.RetrievalPoseFixDataset', 
    train=dict(
        # samples_per_gpu=512,
        # finetune on human annotated data
        samples_per_gpu=32,
        workers_per_gpu=16,
        split='train-val',
        split='train', 
        dataset_root='data/posefix', 
        # finetune on human annotated data
        caption_selection_id=3, 
        stage='finetune', 
        ann_file='processed_dataset/pose_dataset/posefix.pkl', 
    ), 
    val=dict(
        samples_per_gpu=32,
        workers_per_gpu=16,
        split='test',
        dataset_root='data/posefix', 
        # finetune on human annotated data
        caption_selection_id=3, 
        stage='finetune', 
        ann_file='processed_dataset/pose_dataset/posefix.pkl', 
    ), 
    test=dict(
        samples_per_gpu=32,
        workers_per_gpu=16,
        split='test',
        dataset_root='data/posefix', 
        # finetune on human annotated data
        caption_selection_id=3, 
        stage='finetune', 
        ann_file='processed_dataset/pose_dataset/posefix.pkl', 
    ),
)
