_base_ = ['./posegpt_base.py']
# ====================================================
# Dataset: 只做生成任务，不做理解任务
# ====================================================

data=dict(
    train=dict(
        image2pose_dataset=dict(
            split='train',
            dataset_root='processed_dataset/image_dataset', 
            dataset_list=[['mpi-inf', 0.04], ['h36m', 0.1], ['mpii', 0.1], ['coco', 0.1]], 
            data_length=100_000, 
            task_name='image2pose', 
        ), 
        image2text_dataset=dict(
            split='train', 
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/image-posescript-train.pkl', 
            # task_name='image2text', 
        ), 
        image_difference_dataset=dict(
            split='train', 
            dataset_root='processed_dataset/image_dataset', 
            ann_file='processed_dataset/image_dataset/image-posefix-train.pkl', 
            # task_name='image_difference', 
        ), 
        text2pose_dataset=dict(
            split='train', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            # task_name='text2pose', 
        ), 
        pose2text_dataset=dict(
            split='train', 
            dataset_root='data/posescript', 
            ann_file='processed_dataset/pose_dataset/posescript.pkl', 
            # task_name='pose2text', 
        ), 
        pose_edit_dataset=dict(
            split='train', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            # task_name='pose_edit', 
        ), 
        pose_difference_dataset=dict(
            split='train', 
            dataset_root='data/posefix', 
            ann_file='processed_dataset/pose_dataset/posefix.pkl',
            # task_name='pose_difference', 
        ), 
    ), 
)
