import sys
sys.path.append('/home/human/codes/liyiheng/codes/posegpt')
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import torch
import torch.utils
from posegpt.utils import Config
from posegpt.datasets.posegpt import build_data_module
from posegpt.models.base_module import build_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from posegpt.models.posegpt_mlp import PoseGPTMLP
from llava import conversation as conversation_lib
from transformers import AutoTokenizer
from llava.model.language_model.llava_mistral import LlavaMistralConfig
import warnings
from accelerate import Accelerator
import json
import torch.distributed as dist
# debug
# from processed_dataset.vis_utils import vis_mesh
# import cv2

def load_pretrained_model(config, model_path, model_base, device_map='auto', torch_dtype=None, **kwargs):
    # load tokenizer
    assert 'checkpoint' in model_path
    if model_path.endswith('/'): model_path = model_path[:-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # load model
    print('Loading LLaVA from base model...')
    lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
    # NOTE: 初始化vision tower的时候会报warning: 
    # NOTE: copying from a non-meta parameter in the checkpoint to a meta parameter, 官方说没事
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = PoseGPTMLP.from_pretrained(
            model_base, 
            low_cpu_mem_usage=True, 
            attn_implementation=None, 
            torch_dtype=torch_dtype, 
            config=lora_cfg_pretrained, 
            tokenizer=tokenizer, 
            device_map=device_map, 
            pose_vqvae_codebook_size=1, 
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
    model, image_processor = load_pretrained_model(
        config, args.model_path, args.model_base, torch_dtype=torch_dtype, device_map={"": local_rank}, **config)

    # build dataset
    data_module = build_data_module(eval_dataset_config=config.data.eval, image_processor=image_processor, debug=True)
    dataset = data_module['eval_dataset']
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, 
                            collate_fn=data_module['data_collator'])
    accelerator = Accelerator()
    dataloader, model = accelerator.prepare(dataloader, model)
    model.eval()
    
    # build metrics
    all_outputs = []
    for i, data in enumerate(tqdm(dataloader)):
        # move data to cuda
        data = {k: v.to(accelerator.device, dtype=torch_dtype) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        data.pop('imgA_path')
        # print(data.pop('imgA_path'))
        data.pop('imgB_path')

        # forward
        with torch.no_grad():
            all_outputs.append(model.evaluate(**data))
        
        # if i > 100: break

    # gather data
    all_gathered_output = [None for _ in range(accelerator.num_processes)]
    if dist.is_initialized():
        dist.all_gather_object(all_gathered_output, all_outputs)

    # save results
    if accelerator.is_main_process:
        metric = build_model(config.metrics[config.evaluate_task]).eval().cpu()
        for gathered in all_gathered_output:
            for output in gathered:
                metric.update(**{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()})
        result = metric.compute()
        print(result)
        # with open(os.path.join(args.model_path, f'result_{config.evaluate_task}.json'), 'w') as f:
        #     result = {k: float(v) for k, v in result.items()}
        #     json.dump(result, f)

if __name__ == "__main__":
    import random
    import numpy as np
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--bf16', action='store_true', required=True)
    args = parser.parse_args()

    eval_model(args)
