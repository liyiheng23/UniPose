import sys
sys.path.append('/home/human/codes/liyiheng/codes/posegpt')
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import transformers
import torch
import os
from peft.peft_model import PeftModel
from posegpt.models.posegpt_full_mask import PoseGPTFullMaskOnlyPoseVit
from posegpt.datasets.posegpt import build_data_module
from llava.train.train import (ModelArguments, LLaVATrainer, 
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3)
from posegpt.utils import Config
from llava import conversation as conversation_lib
from posegpt.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN # , IMAGEA_TOKEN, IMAGEB_TOKEN
from scripts.gpt_train import SaveCallback, DataArguments, CustomTrainingArguments, find_all_linear_names

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def load_tokenizer(model_args, training_args, codebook_size):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    # Add pose & image tokens
    tokenizer.add_tokens([f'<pose_id_{i}>' for i in range(codebook_size)])
    # pose begin & end token
    tokenizer.add_tokens([f'<pose_id_{codebook_size}>', f'<pose_id_{codebook_size + 1}>'], special_tokens=True)
    # use img start and end token
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    tokenizer.add_tokens([f'<pose_query_{i}>' for i in range(80)]) # num_tokens = 80
    return tokenizer

def load_model(config, model_args, training_args, data_args, tokenizer, codebook_size, attn_implementation):
    model = PoseGPTFullMaskOnlyPoseVit.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        # extra kwargs
        tokenizer=tokenizer,
        ignore_mismatched_sizes=True,
        pose_vqvae_codebook_size=codebook_size, 
        mm_hidden_size=1280) # hmr vit 1280
    model.config.use_cache = False
    model.config.mm_hidden_size = 1280
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.img_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
    model.config.img_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # build vision branch
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    # * build extra model
    model.get_model().load_pose_vqvae(**config)
    model.get_model().load_hmr_vit_backbone(**config)
    # set device and dtype
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model.get_vision_tower().to(dtype=dtype, device=training_args.device)
    model.get_mm_projector().to(dtype=dtype, device=training_args.device)
    model.get_pose_vqvae().to(dtype=dtype, device=training_args.device)
    model.get_hmr_vit_backbone().to(dtype=dtype, device=training_args.device)

    model.requires_grad_(False)
    model.resize_token_embeddings(len(tokenizer))
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    return model

def train(attn_implementation=None):
    # * process args
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config = Config.fromfile(training_args.config)
    # training_args.dataloader_num_workers = 0
    local_rank = training_args.local_rank
    assert training_args.bits == 16
    conversation_lib.default_conversation = conversation_lib.conv_templates['mistral_instruct']

    # * build tokenizer
    codebook_size = config.pose_vqvae_config.params.quantizer.params.nb_code
    tokenizer = load_tokenizer(model_args, training_args, codebook_size)

    # * build model
    model = load_model(config, model_args, training_args, data_args, tokenizer, codebook_size, attn_implementation)

    train_llm = training_args.train_llm
    train_mm_projector = training_args.train_mm_projector
    train_hmr_vit = training_args.train_hmr_vit
    load_from = training_args.load_from

    if training_args.lora_enable:
        # * stage 1, only pretrain llm
        # if train_llm and not train_mm_projector and not load_from and not train_hmr_vit:
        if train_llm and not load_from and not train_hmr_vit:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model, train_llm=True),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM")
            if training_args.bf16: model.to(torch.bfloat16)
            if training_args.fp16: model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        elif load_from:
            assert 'checkpoints' in training_args.load_from
            print('Loading additional LLaVA weights...')
            non_lora_trainables = torch.load(os.path.join(training_args.load_from, 'non_lora_trainables.bin'), map_location='cpu')
            non_lora_trainables = {(k[len('base_model.model.'):] if k.startswith('base_model.model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)
            print('Loading LoRA weights...')
            # * stage 2, only pretrain mm_projector
            if not train_llm and train_mm_projector:
                model = PeftModel.from_pretrained(model, training_args.load_from)
            # * stage 3, finetune mm_projecror and llm
            elif train_llm and train_mm_projector and load_from and not train_hmr_vit:
                model = PeftModel.from_pretrained(model, training_args.load_from, is_trainable=True)
            elif train_llm and train_mm_projector and load_from and train_hmr_vit:
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # * unfreeze lm_head and embd_tokens
    if train_llm:
        for n, p in model.named_parameters():
            if any([x in n for x in ["lm_head", 'embed_tokens']]):
                p.requires_grad_(True)

    # * unfreeze mm_projector
    if train_mm_projector:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad_(True)

    # ! train hmr, for ablation
    # for n, p in model.named_parameters():
    #     if any([x in n for x in ["hmr"]]):
    #         p.requires_grad_(True)

    # find saved params: get_peft_state_non_lora_maybe_zero_3(model.named_parameters()).keys()
    # [n for n, p in model.named_parameters() if p.requires_grad]

    # * build dataset
    image_processor = model.get_vision_tower().image_processor
    data_module = build_data_module(train_dataset_config=config.data.train, image_processor=image_processor)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args,
                           callbacks=[SaveCallback()], **data_module)

    trainer.train()
    trainer.save_state()
    # * save checkpoint
    checkpoint_dir = os.path.join(training_args.output_dir, 'checkpoint-{}'.format(trainer.state.global_step))
    tokenizer.save_pretrained(checkpoint_dir)
    if training_args.lora_enable:
        model.config.use_cache = True
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters())
        if training_args.local_rank in [-1, 0]:
            model.config.save_pretrained(checkpoint_dir)
            model.save_pretrained(checkpoint_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(checkpoint_dir, 'non_lora_trainables.bin'))
        model.config.use_cache = False
    # * save config
    config.dump(os.path.join(training_args.output_dir, os.path.basename(training_args.config)))

if __name__ == "__main__":
    train()
