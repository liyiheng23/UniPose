_base_ = ['./posegpt_base.py']
# ====================================================
# Dataset
# ====================================================
instruction_finetune = True
# instruction_finetune = False
data=dict(
    train=dict(
        image2pose_reasoning_dataset=dict(
            split='train',
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/reasoning-train.pkl', 
            task_name='image2pose_reasoning', 
            instruction_finetune=True, 
        ), 
        image2text_reasoning_dataset=dict(
            split='train',
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/reasoning-train.pkl', 
            task_name='image2text_reasoning', 
            instruction_finetune=True, 
        ), 
        image2pose_dataset=dict(
            split='train',
            dataset_root='processed_dataset/image_dataset', 
            dataset_list=[['mpi-inf', 0.04], ['h36m', 0.1], ['mpii', 0.1], ['coco', 0.1]], 
            data_length=6250, 
            # task_name='image2pose', 
            instruction_finetune=instruction_finetune, 
        ), 
        image2text_dataset=dict(
            split='train', 
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/image-posescript-train.pkl', 
            # task_name='image2text', 
            stage='finetune', 
            caption_selection_id=3, 
            instruction_finetune=instruction_finetune, 
        ), 
        image_difference_dataset=dict(
            split='train', 
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/image-posefix-train.pkl', 
            # task_name='image_difference', 
            stage='finetune', 
            caption_selection_id=3, 
            instruction_finetune=instruction_finetune, 
        ), 
        text2pose_dataset=dict(
            split='train-val', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            stage='finetune', 
            caption_selection_id=3, 
            # task_name='text2pose', 
            instruction_finetune=instruction_finetune, 
        ), 
        pose2text_dataset=dict(
            split='train-val', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            stage='finetune', 
            caption_selection_id=3, 
            # task_name='pose2text', 
            instruction_finetune=instruction_finetune, 
        ), 
        pose_edit_dataset=dict(
            split='train-val', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            # task_name='pose_edit', 
            stage='finetune', 
            caption_selection_id=3, 
            instruction_finetune=instruction_finetune, 
        ), 
        pose_difference_dataset=dict(
            split='train-val', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            # task_name='pose_difference', 
            stage='finetune', 
            caption_selection_id=3, 
            instruction_finetune=instruction_finetune, 
        ), 
    ), 
)
evaluate_task='text2pose'