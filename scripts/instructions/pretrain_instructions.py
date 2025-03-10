from posegpt.constants import IMAGE_TOKEN, POSE_TOKEN, CAPTION_TOKEN, POSEA_TOKEN, POSEB_TOKEN # , IMAGEA_TOKEN, IMAGEB_TOKEN
from easydict import EasyDict

__all__ = [
    'image_dataset_instructions', 'posescript_dataset_instructions', 'posefix_dataset_instructions', 
    'all_task_names']

# by task
instructions = EasyDict(
    image2pose=dict(
        input=[f'Generate pose of the image: {IMAGE_TOKEN}.'], 
        output=[f'{POSE_TOKEN}.']
    ), 

    image2text=dict(
        input=[f'Generate description of the pose in the image: {IMAGE_TOKEN}.'], 
        output=[f'{CAPTION_TOKEN}.']
    ), 

    image_difference=dict(
        input=[f"Output the difference between image {IMAGE_TOKEN} and {IMAGE_TOKEN}."], 
        output=[f'{CAPTION_TOKEN}.']
    ), 

    text2pose=dict(
        # text to pose
        input=[f"Generate pose of the description: {CAPTION_TOKEN}."], 
        output=[f'{POSE_TOKEN}.']
    ), 

    pose2text=dict(
        # pose to text
        input=[f"Generate description of the pose {POSE_TOKEN}."], 
        output=[f'{CAPTION_TOKEN}.']
    ),

    pose_difference=dict(
        input=[f"Output the difference between pose {POSEA_TOKEN} and pose {POSEB_TOKEN}."], 
        output=[f'{CAPTION_TOKEN}.']
    ), 

    pose_edit=dict(
        input=[f"Edit pose {POSEA_TOKEN} by description: {CAPTION_TOKEN}."], 
        output=[f'{POSEB_TOKEN}.']
    )
)