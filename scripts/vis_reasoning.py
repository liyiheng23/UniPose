import sys
sys.path.append('/home/human/codes/liyiheng/codes/posegpt')
import os
import argparse
import torch
import torch.utils
from posegpt.utils import Config
from posegpt.datasets.posegpt import build_data_module
from torch.utils.data import DataLoader
from tqdm import tqdm
from llava import conversation as conversation_lib
from scripts.gpt_eval_full_mask import load_pretrained_model
from posegpt.utils.vis_for_tasks import render_smpl, get_smpl_pose_params
import cv2
from posegpt.models.metrics import NLPEvaluator
import numpy as np

device = 'cuda:0'
def eval_model(args):
    # disable_torch_init()
    config = Config.fromfile(args.config)
    torch_dtype = torch.bfloat16
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    conversation_lib.default_conversation = conversation_lib.conv_templates['mistral_instruct']
    
    # build model, tokenizer 
    print('Load model...')
    model, image_processor = load_pretrained_model(
        config, args.model_path, args.model_base, torch_dtype=torch_dtype, device_map={"": local_rank}, **config)

    # build dataset
    print('Load data...')
    data_module = build_data_module(eval_dataset_config=config.data.eval, image_processor=image_processor, debug=True)
    dataset = data_module['eval_dataset']
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, 
                            collate_fn=data_module['data_collator'])
    nlp_evaluator = NLPEvaluator(metric_list=['bleu', 'rouge', 'meteor'])
    # print(f'&&&&&&&&&&&&&&&{config.evaluate_task}&&&&&&&&&&&&&&&&')
    for i, data in enumerate(tqdm(dataloader)):
        # if i < 1900: continue
        # move data to cuda
        data = {k: v.to(device, dtype=torch_dtype) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        imgA_path = data.pop('imgA_path', None)
        imgB_path = data.pop('imgB_path', None)
        ori_image = data.pop('ori_image', None)
        # forward
        with torch.no_grad(): output = model.evaluate(**data)

        # output result
        print(data['tasks'][0])
        print(data['caption'])
        if config.evaluate_task == 'image2pose_reasoning':
            # change posegpt/utils/vis_for_tasks.py # 125 line
            # import ipdb; ipdb.set_trace()The leftmost person in the picture.
            # data['tasks'][0]['input'] = "The leftmost person in the picture. The person is wearing white vest and gray skirt. Take a look at the image <image> and return the SMPL pose parameters for the figure shown."

            # data['tasks'][0]['input'] = "On the image's right side, this person is visible. The person is wearing a light blue t-shirt and blue jeans. The left arm is forward, forming an L-shape, while the right arm is lower and bent, with hands wide apart. In the image <image>, please analyze the SMPL pose of the person you see."
            # with torch.no_grad(): output = model.evaluate(**data) 
            
            cv2.imwrite('gt_img.png', ori_image[0].to(torch.uint8).cpu().numpy()[..., ::-1])

            render_smpl(get_smpl_pose_params(data['body_poseA_rotmat'], type='rotmat', normalize_root_orient=False), save_path='gt_smpl.png', viewpoints=[[-45, (0, 1, 0)]])

            render_smpl(get_smpl_pose_params(output['pred_axis_angles'].reshape(1, -1, 3), type='axis_angle', normalize_root_orient=False), save_path='pred_smpl.png', viewpoints=[[-45, (0, 1, 0)]])
        elif config.evaluate_task == 'image2text_reasoning':
            # change posegpt/utils/vis_for_tasks.py # 125 line
            # i == 1905
            cv2.imwrite('gt_img.png', cv2.imread(imgA_path[0]))
            render_smpl(get_smpl_pose_params(data['body_poseA_rotmat'], type='rotmat', normalize_root_orient=False), 
                        save_path='gt_smpl.png', viewpoints=[[-45, (0, 1, 0)]])
            print(output['pred_text'])
        else:
            raise NotImplementedError
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default='checkpoints/stage-finetune-full-mask-instructions-fix-hmr-bug-image-reasoning-epoch-12/checkpoint-5472')
    parser.add_argument("--model-base", type=str, default='cache/llava-v1.6-mistral-7b')
    parser.add_argument("--config", type=str, default='configs/posegpt_eval_finetune_reasoning.py')
    parser.add_argument('--bf16', default=True)
    args = parser.parse_args()

    eval_model(args)
