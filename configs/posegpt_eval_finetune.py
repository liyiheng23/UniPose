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
            # dataset_list=[['h36m_val', 1.0]], 

            training=False, 
            instruction_finetune=False, 
            # task_name='image2pose', 
        ), 
        image2text_dataset=dict(
            split='test', 
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/image-posescript-test.pkl', 
            instruction_finetune=True, 
            # task_name='image2text', 
            caption_selection_id=3, 
        ), 
        image_difference_dataset=dict(
            split='test', 
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/image-posefix-test.pkl', 
            instruction_finetune=True, 
            # task_name='image_difference', 
            caption_selection_id=3, 
        ), 
        text2pose_dataset=dict(
            split='test', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            stage='finetune',
            instruction_finetune=False, 
            caption_selection_id=0, 
            # task_name='text2pose', 
        ), 
        pose2text_dataset=dict(
            split='test', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            stage='finetune',
            caption_selection_id=3, 
            instruction_finetune=False, 
            task_name='pose2text', 
        ), 
        pose_edit_dataset=dict(
            split='test', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            stage='finetune',
            caption_selection_id=3, 
            instruction_finetune=True, 
            # task_name='pose_edit', 
        ), 
        pose_difference_dataset=dict(
            split='test', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            stage='finetune',
            caption_selection_id=3, 
            instruction_finetune=True, 
            # task_name='pose_difference', 
        ), 
    ), 
)

for dataset in data['eval'].values():
    if 'task_name' in dataset:
        evaluate_task = dataset['task_name']