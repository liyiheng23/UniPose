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
            task_name='image2text', 
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
            stage='pretrain',
            caption_selection_id=0, 
            # task_name='text2pose', 
        ), 
        pose2text_dataset=dict(
            split='test', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            stage='pretrain',
            caption_selection_id=0, 
            # task_name='pose2text', 
        ), 
        pose_edit_dataset=dict(
            split='test', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            stage='pretrain',
            caption_selection_id=0, 
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

for dataset in data['eval'].values():
    if 'task_name' in dataset:
        evaluate_task = dataset['task_name']
