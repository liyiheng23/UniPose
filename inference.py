import os
import cv2
import argparse
import torch
import torch.utils
from tqdm import tqdm
from llava import conversation as conversation_lib
from transformers import AutoTokenizer
from llava.model.language_model.llava_mistral import LlavaMistralConfig
from peft import PeftModel
from torch.ut

from posegpt.utils import Config
from posegpt.models.posegpt_full_mask import PoseGPTFullMask
from posegpt.constants import POSE_TOKEN, IMAGE_TOKEN

def load_pretrained_model(config, model_path, model_base, device_map='auto', torch_dtype=None, **kwargs):
    # load tokenizer
    assert 'checkpoint' in model_path
    if model_path.endswith('/'): model_path = model_path[:-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # load model
    print('Loading LLaVA from base model...')
    lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
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

def hmr_transform(n_px=256):
    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), 
                  (0.229, 0.224, 0.225)),
    ])

device = 'cuda:0'
def main(args):
    # disable_torch_init()
    config = Config.fromfile(args.config)
    torch_dtype = torch.bfloat16
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    conversation_lib.default_conversation = conversation_lib.conv_templates['mistral_instruct']
    
    # build model, tokenizer 
    print('Load model...')
    model, image_processor = load_pretrained_model(
        config, args.model_path, args.model_base, torch_dtype=torch_dtype, device_map={"": local_rank}, **config)
    
    print("Use '<image>' as the image placeholder, and '<pose>' as the pose placeholder. Here are some examples: ")
    print("Example 1: Generate pose of the image <image>.")
    print("Example 2: Output the difference between <pose> and <pose>.")
    print('Example 3: Output the difference between <image> and <image>.')

    
    while True:
        prompt = input('User:')

        task = dict(input=prompt)
        poseA_rotmat_path = None
        poseB_rotmat_path = None
        imgA_path = None
        imgB_path = None

        if prompt.count(POSE_TOKEN) == 1:
            poseA_rotmat_path = input('Input file path of the pose (in rotmat):')
        if prompt.count(POSE_TOKEN) == 2:
            poseA_rotmat_path = input('Input file path of the pose A (in rotmat):')
            poseB_rotmat_path = input('Input file path of the pose B (in rotmat):')
        if prompt.count(IMAGE_TOKEN) == 1:
            imgA_path = input('Input file path of the image:')
        if prompt.count(IMAGE_TOKEN) == 2:
            imgA_path = input('Input file path of the image A:')
            imgB_path = input('Input file path of the image B:')
        
        if imgA_path 




        

    print(f'&&&&&&&&&&&&&&&{config.evaluate_task}&&&&&&&&&&&&&&&&')
    for i, data in enumerate(tqdm(dataloader)):
        # if i < 1900: continue
        # move data to cuda
        data = {k: v.to(device, dtype=torch_dtype) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        import ipdb; ipdb.set_trace()
        imgA_path = data.pop('imgA_path', None)
        imgB_path = data.pop('imgB_path', None)
        if '3dpw' in imgB_path[0] or 'h36m' in imgB_path[0]:
            continue
        # forward
        with torch.no_grad(): output = model.evaluate(**data)

        # output result
        print(data['tasks'][0])
        print(data['caption'])
        if config.evaluate_task == 'pose2text':
            print(output['pred_text'])
            gt_pose_params = get_smpl_pose_params(output['gt_pose'], type='rotmat')
            render_smpl(gt_pose_params, save_path='gt_smpl.png')
        elif config.evaluate_task == 'text2pose':
            # forward
            with torch.no_grad(): output = model.evaluate(**data)
            # output result
            render_smpl(get_smpl_pose_params(output['gt_pose'], type='rotmat'), save_path='gt_smpl.png')
            render_smpl(get_smpl_pose_params(output['pred_pose'], type='rotmat'), save_path='pred_smpl.png', viewpoints=[[20, (0, 1, 0)]])
        elif config.evaluate_task == 'pose_difference':
            gt_pose_A, gt_pose_B = output['gt_pose']
            render_smpl(get_smpl_pose_params(gt_pose_A, type='rotmat'), save_path='gt_smpl_a.png', viewpoints=[[20, (0, 1, 0)]])
            render_smpl(get_smpl_pose_params(gt_pose_B, type='rotmat'), save_path='gt_smpl_b.png', viewpoints=[[20, (0, 1, 0)]])
            print(output['pred_text'])
        elif config.evaluate_task == 'pose_edit':            
            render_smpl(get_smpl_pose_params(data['body_poseA_rotmat'], type='rotmat'), save_path='gt_smpl_a.png', viewpoints=[[20, (0, 1, 0)]])
            # import ipdb; ipdb.set_trace()
            # data['caption'][0] = 'Put down your left leg and stretch it as straight as possible. Extend your right arm forward and your left arm backward'
            # with torch.no_grad(): output = model.evaluate(**data)
            render_smpl(get_smpl_pose_params(output['gt_pose'], type='rotmat'), save_path='gt_smpl_b.png', viewpoints=[[20, (0, 1, 0)]])
            render_smpl(get_smpl_pose_params(output['pred_pose'], type='rotmat'), save_path='pred_smpl_b.png', viewpoints=[[20, (0, 1, 0)]])
        elif config.evaluate_task == 'image2pose':
            # 右手向后伸，与身体呈45度，
            # change posegpt/utils/vis_for_tasks.py # 125 line
            # data['tasks'][0]['input'] = 'The right elbow is straight. Extend your right hand as far back as possible. The right arm is located behind the body.  The left knee partially bent while the left forearm is aligned horizontally. Can you examine the image <image> and identify the SMPL pose parameters of the individual?'
            # with torch.no_grad(): output = model.evaluate(**data) 
            vis_mesh(output['pred_axis_angles'][0].float().cpu())
            import ipdb; ipdb.set_trace()

            # cv2.imwrite('gt_img.png', cv2.imread(imgA_path[0]))
            render_smpl(get_smpl_pose_params(data['body_poseA_rotmat'], type='rotmat', normalize_root_orient=False), save_path='gt_smpl.png', viewpoints=[[-45, (0, 1, 0)]])

            render_smpl(get_smpl_pose_params(output['pred_axis_angles'].reshape(1, -1, 3), type='axis_angle', normalize_root_orient=False), save_path='pred_smpl.png', viewpoints=[[-45, (0, 1, 0)]])
        elif config.evaluate_task == 'image2text':
            # change posegpt/utils/vis_for_tasks.py # 125 line
            # i == 1905
            cv2.imwrite('gt_img.png', cv2.imread(imgA_path[0]))
            render_smpl(get_smpl_pose_params(data['body_poseA_rotmat'], type='rotmat', normalize_root_orient=False), 
                        save_path='gt_smpl.png', viewpoints=[[-45, (0, 1, 0)]])
            print(output['pred_text'])
            metric = nlp_evaluator(predictions=output['pred_text'], references=output['gt_text'])
            print(f"bleu: {metric['bleu']['bleu']}, rouge: {metric['rouge']['rougeL']}, meteor: {metric['meteor']['meteor']}")
        elif config.evaluate_task == 'image_difference':
            # imgA_path = 'processed_dataset/image_dataset/3dpw/imageFiles_outdoors_fencing_01_image_00917.png'
            # imgB_path = 'processed_dataset/image_dataset/3dpw/imageFiles_outdoors_fencing_01_image_00648.png'

            # imageA = cv2.cvtColor(cv2.imread(imgA_path), cv2.COLOR_BGR2RGB)
            # imageB = cv2.cvtColor(cv2.imread(imgB_path), cv2.COLOR_BGR2RGB)
            # imageA = dataloader.dataset.datasets[2].image_processor.preprocess(imageA, return_tensors='pt')['pixel_values'][0]
            # imageB = dataloader.dataset.datasets[2].image_processor.preprocess(imageB, return_tensors='pt')['pixel_values'][0]
            # data['images'] = torch.stack([imageA, imageB]).cuda().bfloat16()
            cv2.imwrite('gt_img_a.png', cv2.imread(imgA_path[0]))
            cv2.imwrite('gt_img_b.png', cv2.imread(imgB_path[0]))
            gt_pose_A, gt_pose_B = output['gt_pose']
            render_smpl(get_smpl_pose_params(gt_pose_A, type='rotmat', normalize_root_orient=False), 
                        save_path='gt_smpl_a.png', viewpoints=[[-45, (0, 1, 0)]])
            render_smpl(get_smpl_pose_params(gt_pose_B, type='rotmat', normalize_root_orient=False), 
                        save_path='gt_smpl_b.png', viewpoints=[[-45, (0, 1, 0)]])
            metric = nlp_evaluator(predictions=output['pred_text'], references=output['gt_text'])
            print(f"bleu: {metric['bleu']['bleu']}, rouge: {metric['rouge']['rougeL']}, meteor: {metric['meteor']['meteor']}")
            print(output['pred_text'])
        else:
            continue
        # if metric['bleu']['bleu'] > 0.15:
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default='checkpoints/stage-finetune-full-mask-instructions-fix-hmr-bug-epoch-6/checkpoint-7260')
    parser.add_argument("--model-base", type=str, default='cache/llava-v1.6-mistral-7b')
    parser.add_argument("--config", type=str, default='configs/vis.py')
    parser.add_argument('--bf16', default=True)
    args = parser.parse_args()

    main(args)
