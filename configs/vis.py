_base_ = ['./posegpt_base.py']
# ====================================================
# Dataset
# ====================================================

instruction_finetune = True
# instruction_finetune = False
data=dict(
    eval=dict(
        image2pose_reasoning_dataset=dict(
            split='train',
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/reasoning-test.pkl', 
            # task_name='image2pose_reasoning', 
            instruction_finetune=True, 
        ), 
        image2text_reasoning_dataset=dict(
            split='train',
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/reasoning-test.pkl', 
            # task_name='image2text_reasoning', 
            instruction_finetune=True, 
        ), 
        image2pose_dataset=dict(
            split='test',
            dataset_root='processed_dataset/image_dataset', 
            
            dataset_list=[['lsp', 1.0]],
            # dataset_list=[['h36m_val', 1.0]], 

            training=False, 
            task_name='image2pose', 

            instruction_finetune=instruction_finetune, 
        ), 
        image2text_dataset=dict(
            split='test', 
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/image-posescript-test.pkl', 
            # task_name='image2text', 
            # task_name = 'image2pose', 
            caption_selection_id=0, 
            instruction_finetune=instruction_finetune, 
        ), 
        image_difference_dataset=dict(
            split='test', 
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/image-posefix-test.pkl', 
            # task_name='image_difference', 
            caption_selection_id=3, 
            instruction_finetune=instruction_finetune, 
        ), 
        text2pose_dataset=dict(
            split='test', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            stage='finetune',
            caption_selection_id=3, 
            # task_name='text2pose', 
            instruction_finetune=instruction_finetune, 
        ), 
        pose2text_dataset=dict(
            split='test', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            stage='finetune',
            caption_selection_id=3, 
            # task_name='pose2text', 
            instruction_finetune=instruction_finetune, 
        ), 
        pose_edit_dataset=dict(
            split='test', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            stage='finetune',
            caption_selection_id=3, 
            # task_name='pose_edit', 
            instruction_finetune=instruction_finetune, 
        ), 
        pose_difference_dataset=dict(
            split='test', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            stage='finetune',
            caption_selection_id=3, 
            # task_name='pose_difference', 
            instruction_finetune=instruction_finetune, 
        ), 
    ), 
)

for dataset in data['eval'].values():
    if 'task_name' in dataset:
        evaluate_task = dataset['task_name']