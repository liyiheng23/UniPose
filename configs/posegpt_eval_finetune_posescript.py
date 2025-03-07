_base_ = ['./posegpt_base.py']
# ====================================================
# Dataset
# ====================================================

data=dict(
    eval=dict(
        image2pose_dataset=dict(
            split='test',
            dataset_root='processed_dataset/image_dataset', 
            dataset_list=[['3dpw', 1.0]], 
            # task_name='image2pose', 
        ), 
        image2text_dataset=dict(
            split='test', 
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/image-posescript-test.pkl', 
            # task_name='image2text', 
        ), 
        image_difference_dataset=dict(
            split='test', 
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/image-posefix-test.pkl', 
            # task_name='image_difference', 
        ), 
        text2pose_dataset=dict(
            split='test', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            stage='finetune',
            caption_selection_id=3, 
            instruction_finetune=True, 
            task_name='text2pose', 
        ), 
        pose2text_dataset=dict(
            split='test', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            stage='finetune',
            caption_selection_id=3, 
            # task_name='pose2text', 
        ), 
        pose_edit_dataset=dict(
            split='test', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            stage='finetune',
            caption_selection_id=3, 
            # task_name='pose_edit', 
        ), 
        pose_difference_dataset=dict(
            split='test', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            stage='finetune',
            caption_selection_id=3, 
            # task_name='pose_difference', 
        ), 
    ), 
)


# ====================================================
# Metric
# ====================================================

pose_text_encoder_config=dict(
    target='posegpt.models.posescript_retrieval.PoseText', 
    params=dict(), 
)

poseAB_text_encoder_config=dict(
    target='posegpt.models.posescript_retrieval.PairText', 
    params=dict(), 
)

pose_text_encoder_ckp_path = 'checkpoints/ret_distilbert_dataPSA2ftPSH2/seed1/checkpoint_best.pth'
poseAB_text_encoder_ckp_path = 'checkpoints/modret_distilbert_dataPFAftPFH/seed1/checkpoint_best.pth'

metrics = dict(
    text2pose=dict(
        target='posegpt.models.posescript_metrics.Text2PoseMetric', 
        params=dict(
            pose_text_encoder_config=pose_text_encoder_config, 
            pose_text_encoder_ckp_path=pose_text_encoder_ckp_path)
    ), 
    pose2text=dict(
        target='posegpt.models.posescript_metrics.Pose2TextMetric', 
        params=dict(
            pose_text_encoder_config=pose_text_encoder_config, 
            pose_text_encoder_ckp_path=pose_text_encoder_ckp_path)
    ), 
    image2text=dict(
        target='posegpt.models.posescript_metrics.Pose2TextMetric', 
        params=dict(
            pose_text_encoder_config=pose_text_encoder_config, 
            pose_text_encoder_ckp_path=pose_text_encoder_ckp_path)
    ), 
    image_difference=dict(
        target='posegpt.models.posescript_metrics.PoseDifferenceMetric', 
        params=dict(
            pose_text_encoder_config=poseAB_text_encoder_config, 
            pose_text_encoder_ckp_path=poseAB_text_encoder_ckp_path
        ), 
    ), 
    image2pose=dict(
        target='posegpt.models.posescript_metrics.PoseReconstructionMetric', 
        params=dict(), 
    ), 
    pose_edit=dict(
        target='posegpt.models.posescript_metrics.PoseEditMetric', 
        params=dict(
            pose_text_encoder_config=pose_text_encoder_config, 
            pose_text_encoder_ckp_path=pose_text_encoder_ckp_path
        ), 
    ), 
    pose_difference=dict(
        target='posegpt.models.posescript_metrics.PoseDifferenceMetric', 
        params=dict(
            pose_text_encoder_config=poseAB_text_encoder_config, 
            pose_text_encoder_ckp_path=poseAB_text_encoder_ckp_path
        ), 
    ), 
)

for dataset in data['eval'].values():
    if 'task_name' in dataset:
        evaluate_task = dataset['task_name']