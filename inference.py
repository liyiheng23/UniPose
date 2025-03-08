import sys
import os
import cv2
import argparse
import torch
import torch.utils
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from llava import conversation as conversation_lib

from posegpt.utils.vis_for_tasks import render_smpl, get_smpl_pose_params
from posegpt.utils import Config
from posegpt.datasets.posegpt import build_data_module

device = 'cuda:1'

def load_pretrained_model(config, model_path, model_base, device_map='auto', torch_dtype=None, **kwargs):
    # load tokenizer
    assert 'checkpoint' in model_path
    if model_path.endswith('/'): model_path = model_path[:-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # load model
    print('Loading LLaVA from base model...')
    lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = PoseGPTFullMask.from_pretrained(
            model_base, 
            low_cpu_mem_usage=True, 
            attn_implementation=None, 
            torch_dtype=torch_dtype, 
            config=lora_cfg_pretrained, 
            tokenizer=tokenizer, 
            device_map=device_map, 
            pose_vqvae_codebook_size=config.pose_vqvae_config.params.quantizer.params.nb_code, 
            evaluate_task=config.evaluate_task)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    model.model.mm_projector[0].weight = torch.nn.Parameter(torch.empty(4096, 2304, device=model.device, dtype=model.dtype))

    model.get_model().load_hmr_vit_backbone(**config)

    print('Loading additional LLaVA weights...')
    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    non_lora_trainables = {(k[len('base_model.model.'):] if k.startswith('base_model.model.') else k): v for k, v in non_lora_trainables.items()}
    model.resize_token_embeddings(len(tokenizer)) # type: ignore
    model.load_state_dict(non_lora_trainables, strict=False)
    
    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    # build pose vqvae model
    model.get_model().load_pose_vqvae(**config)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        raise NotImplementedError
    image_processor = vision_tower.image_processor
    model.get_pose_vqvae().to(model.device).to(torch_dtype)
    model.get_hmr_vit_backbone().to(model.device).to(torch_dtype)
    return model, image_processor


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
