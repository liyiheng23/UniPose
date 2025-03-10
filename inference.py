import os
import cv2
import argparse
import warnings
import trimesh
import torch
import torch.utils
from tqdm import tqdm
from llava import conversation as conversation_lib
from transformers import AutoTokenizer
from llava.model.language_model.llava_mistral import LlavaMistralConfig
from peft import PeftModel
from torchvision.transforms.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import numpy as np

from posegpt.utils import Config
from posegpt.utils import BodyModel
from posegpt.models.posegpt_full_mask import PoseGPTFullMask
from posegpt.constants import POSE_TOKEN, IMAGE_TOKEN
from posegpt.utils.rotation_conversions import axis_angle_to_matrix

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def load_pretrained_model(config, model_path, model_base, device_map='auto', torch_dtype=None, **kwargs):
    # load tokenizer
    if model_path.endswith('/'): model_path = model_path[:-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # load model
    # print('Loading LLaVA from base model...')
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
            evaluate_task=None)

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

    # print('Loading additional LLaVA weights...')
    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    non_lora_trainables = {(k[len('base_model.model.'):] if k.startswith('base_model.model.') else k): v for k, v in non_lora_trainables.items()}
    model.resize_token_embeddings(len(tokenizer)) # type: ignore
    model.load_state_dict(non_lora_trainables, strict=False)
    
    # print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    # print('Merging LoRA weights...')
    model = model.merge_and_unload()
    # print('Model is loaded...')

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

def vis_mesh(pose=None, pose_body=None, global_orient=None, save_path='smpl_mesh.obj'):
    if pose is not None:
        pose = torch.tensor(pose).to(torch.float64)
        pose_body = pose[None, 3:66]
        root_orient = pose[None, :3]
    else:
        pose_body = pose_body
        root_orient = global_orient
    smpl = BodyModel('cache/smpl_models/smplx/SMPLX_NEUTRAL.npz', dtype=torch.float64)
    p1 = smpl.forward(pose_body=pose_body, root_orient=root_orient)
    trimesh.Trimesh(vertices=p1.v.detach().numpy()[0], faces=smpl.f).export(save_path)

def main(args):
    # disable_torch_init()
    config = Config.fromfile(args.config)
    torch_dtype = torch.bfloat16
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f'cuda:{local_rank}'
    conversation_lib.default_conversation = conversation_lib.conv_templates['mistral_instruct']
    
    # build model, tokenizer 
    print('Load model...')
    model, image_processor = load_pretrained_model(
        config, args.model_path, args.model_base, torch_dtype=torch_dtype, device_map={"": local_rank}, **config)
    hmr_image_processor = hmr_transform(n_px=256)
    
    print("Use '<image>' as the image placeholder, and '<pose>' as the pose placeholder. Here are some examples: ")
    print("Example 1: Generate pose of the image <image>.")
    print("Example 2: Output the difference between <pose> and <pose>.")
    print('Example 3: Output the difference between <image> and <image>.')
    
    while True:
        prompt = input('=> User: ')

        poseA_rotmat_path = None
        poseB_rotmat_path = None
        imgA_path = None
        imgB_path = None

        if prompt.count(POSE_TOKEN) == 1:
            poseA_rotmat_path = input('==>Input file path of the pose (in rotmat): ')
        if prompt.count(POSE_TOKEN) == 2:
            poseA_rotmat_path = input('==>Input file path of the pose A (in rotmat): ')
            poseB_rotmat_path = input('==>Input file path of the pose B (in rotmat): ')
        if prompt.count(IMAGE_TOKEN) == 1:
            imgA_path = input('==>Input file path of the image: ')
        if prompt.count(IMAGE_TOKEN) == 2:
            imgA_path = input('==>Input file path of the image A: ')
            imgB_path = input('==>Input file path of the image B: ')
        
        body_poseA_rotmat = torch.zeros((22, 3, 3))
        body_poseB_rotmat = torch.zeros((22, 3, 3))
        if poseA_rotmat_path is not None:
            body_poseA_rotmat = torch.from_numpy(np.load(poseA_rotmat_path))
        if poseB_rotmat_path is not None:
            body_poseB_rotmat = torch.from_numpy(np.load(poseB_rotmat_path))
        
        imageA = torch.zeros((3, 336, 336))
        imageB = torch.zeros((3, 336, 336))
        hmr_imageA = torch.zeros((3, 256, 256))
        hmr_imageB = torch.zeros((3, 256, 256))
        if imgA_path is not None:
            imageA = cv2.cvtColor(cv2.imread(imgA_path), cv2.COLOR_BGR2RGB)
            imageA = image_processor.preprocess(imageA, return_tensors='pt')['pixel_values'][0]
            hmr_imageA = hmr_image_processor(Image.open(imgA_path))

        if imgB_path is not None:
            imageB = cv2.cvtColor(cv2.imread(imgB_path), cv2.COLOR_BGR2RGB)
            imageB = image_processor.preprocess(imageB, return_tensors='pt')['pixel_values'][0]
            hmr_imageB = hmr_image_processor(Image.open(imgB_path))
        
        batch = dict(
            body_poseA_rotmat=body_poseA_rotmat.to(torch.bfloat16).to(device).unsqueeze(0), 
            body_poseB_rotmat=body_poseB_rotmat.to(torch.bfloat16).to(device).unsqueeze(0),
            images=torch.stack([imageA, imageB], dim=0).to(torch.bfloat16).to(device), 
            hmr_images=torch.stack([hmr_imageA, hmr_imageB], dim=0).to(torch.bfloat16).to(device), 
            tasks=[{'input': prompt}], 
            caption=['']
        )
        
        with torch.no_grad():
            output = model.evaluate(**batch)

        body_pose = output['body_pose']
        text = output['text']
        if text is not None:
            print(f'=> GPT: {text[0]}')
        if body_pose is not None:
            body_pose = body_pose.to(torch.float32).cpu().squeeze(0).numpy()
            np.save('smpl_mesh_rotmat.npy', axis_angle_to_matrix(torch.from_numpy(body_pose).view(-1, 3)))
            vis_mesh(body_pose)
            
            print("SMPL mesh saved as smpl_mesh.obj")
            print("SMPL parameters saved as smpl_mesh_rotmat.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default='cache/unipose')
    parser.add_argument("--model-base", type=str, default='cache/llava-v1.6-mistral-7b')
    parser.add_argument("--config", type=str, default='configs/inference.py')
    args = parser.parse_args()

    main(args)
