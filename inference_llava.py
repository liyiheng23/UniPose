import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import *
import pickle as pkl
import os.path as osp
from tqdm import tqdm

def image_parser(image_file, sep):
    out = image_file.split(sep)
    return out
posescript = pkl.load(open('/home/human/codes/liyiheng/codes/posegpt/processed_dataset/image_dataset/image-posescript-mini-test.pkl', 'rb'))

model_path = "cache/llava-v1.6-mistral-7b"

_prompt = "You are an AI pose analyst tasked with describing the precise physical posture of a person positioned at the center of the image. Focus on providing specific, detailed descriptions of each body part, including the arms, legs, torso and so on. Your description should include angles, directions, and relative positioning to ensure clarity and accuracy. Your output should not exceed 50 words. Only need to output a description of the human body posture. "

model_name = get_model_name_from_path(model_path)
max_new_tokens = 512
num_beams = 1
top_p = None
temperature = 0
sep = ','
conv_mode = None
model_base = None


pose_text_encoder_config=dict(
    target='posegpt.models.retrieval.TextPoseRetrieval', 
    params=dict(
        pose_encoder=dict(
            target='posegpt.models.retrieval.VQPoseEncoder',
            params=dict(
                input_dim=6, # 6d rot
                hidden_dim=512, 
                output_dim=512, 
                upsample_step=1, 
                downsample_step=1,
                num_joints=22)
        ), 

        text_encoder=dict(
            target='posegpt.models.retrieval.TextEncoderBiGRUCo', 
            params=dict(
                hidden_dim=512, 
                output_dim=512)),  
    ), 
)

poseAB_text_encoder_config=dict(
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
                num_joints=22)
        ), 

        text_encoder=dict(
            target='posegpt.models.retrieval.TextEncoderBiGRUCo', 
            params=dict(
                hidden_dim=512, 
                output_dim=512)),  
    ), 
)

pose_text_encoder_ckp_path = 'processed_dataset/saved_checkpoints/retrieval_text_pose_vqpose_finetune_trainval/best_mRecall.ckpt'
poseAB_text_encoder_ckp_path = 'processed_dataset/saved_checkpoints/retrieval_text_poseAB_vqpose_finetune_trainval/best_mRecall.ckpt'

metrics = dict(
    pose2text=dict(
        target='posegpt.models.metrics.Pose2TextMetric', 
        params=dict(
            pose_text_encoder_config=pose_text_encoder_config, 
            pose_text_encoder_ckp_path=pose_text_encoder_ckp_path)
    ), 
    pose_difference=dict(
        target='posegpt.models.metrics.PoseDifferenceMetric', 
        params=dict(
            pose_text_encoder_config=poseAB_text_encoder_config, 
            pose_text_encoder_ckp_path=poseAB_text_encoder_ckp_path
        ), 
    ), 
)

# Model
disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
from posegpt.models.metrics import Pose2TextMetric, PoseDifferenceMetric
from posegpt.utils.rotation_conversions import axis_angle_to_matrix
pose2text_metric = Pose2TextMetric(pose_text_encoder_config, pose_text_encoder_ckp_path).cuda()
posediff_metric = PoseDifferenceMetric(poseAB_text_encoder_config, poseAB_text_encoder_ckp_path).cuda()

import numpy as np
import torch

for i, item in tqdm(enumerate(posescript)):
    qs = _prompt
    image_file = osp.join('processed_dataset/image_dataset', item['img_path'])
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if conv_mode is not None and conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(image_file, sep)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    gt_pose = np.concatenate([item['global_orient'], item['body_pose']], axis=0)
    gt_pose = axis_angle_to_matrix(torch.tensor(gt_pose[:66].reshape(-1, 3)))[None].cuda()
    pose2text_metric.update(gt_pose, [tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()], [item['captions'][-1]])


print(pose2text_metric.compute())