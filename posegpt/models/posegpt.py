import torch
from llava.model import LlavaMistralForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralModel
from posegpt.models.base_module import build_model
from posegpt.utils import load_checkpoint
from posegpt.constants import *
from llava.mm_utils import tokenizer_image_token
from llava import conversation as conversation_lib
import copy
from posegpt.utils import BodyModel
from posegpt.utils.rotation_conversions import matrix_to_axis_angle
import torch.nn.functional as F
from .vit import build_and_load_vit
from transformers import MistralModel
from typing import *

class PoseGPTModel(LlavaMistralModel):
    def load_pose_vqvae(self, pose_vqvae_config, pose_vqvae_ckp_path, **kwargs):
        # load pose vqvae model
        pose_vqvae = build_model(pose_vqvae_config)
        self.pose_vqvae = load_checkpoint(pose_vqvae, pose_vqvae_ckp_path)
        self.pose_vqvae.training = False
        for p in self.pose_vqvae.parameters():
            p.requires_grad = False

    # load vit from tokenhmr
    def load_hmr_vit_backbone(self, hmr_vit_ckp_path, **kwargs):
        self.hmr_vit_backbone = build_and_load_vit(hmr_vit_ckp_path, freeze=True)
    
class PoseGPT(LlavaMistralForCausalLM):
    def __init__(self, config, tokenizer=None, pose_vqvae_codebook_size=None, evaluate_task=None, mm_hidden_size=None):
        # change mm_projector input dim
        if mm_hidden_size is not None:
            config.mm_hidden_size = mm_hidden_size
        super().__init__(config)
        self.model = PoseGPTModel(config)
        self.tokenizer = tokenizer
        self.pose_vqvae_codebook_size = pose_vqvae_codebook_size
        self.pose_begin_idx = self.tokenizer(f'<pose_id_{self.pose_vqvae_codebook_size}>').input_ids[1] # type: ignore
        self.pose_end_idx = self.tokenizer(f'<pose_id_{self.pose_vqvae_codebook_size + 1}>').input_ids[1] # type: ignore
        if 'evaluate_task' is not None:
            self.evaluate_task = evaluate_task

    def get_mm_projector(self):
        return self.model.mm_projector
    
    def get_pose_vqvae(self):
        return self.model.pose_vqvae
    
    def get_hmr_vit_backbone(self):
        return self.model.hmr_vit_backbone
    
    def evaluate(self, 
                body_poseA_rotmat, 
                body_poseB_rotmat, 
                images, 
                caption, 
                tasks, 
                **kwargs):
        # pose tokenizer
        # if self.evaluate_task in ['image2pose']:
        #     poseA_tokens = torch.zeros((body_poseA_rotmat.shape[0], 80)).to(body_poseA_rotmat.device).to(torch.int64)
        #     poseB_tokens = poseA_tokens.clone()
        # else:
        keypoints_3d = kwargs.pop('keypoints_3d', None)
        poseA_tokens = self.model.pose_vqvae.encode(body_poseA_rotmat)
        poseB_tokens = self.model.pose_vqvae.encode(body_poseB_rotmat)
        input_ids, attention_mask = process_templates(
            caption, tasks, poseA_tokens, poseB_tokens, tokenizer=self.tokenizer, 
            codebook_size=self.pose_vqvae_codebook_size)
        
        # generate
        self.config.tokenizer_padding_side = 'left'
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        outputs = self.generate(
            input_ids, attention_mask=attention_mask, images=images, max_new_tokens=512, 
            num_beams=1, use_cache=True, **kwargs)
        # convert output to float32 for metric calculation.
        # pose process
        if self.evaluate_task in ['pose2text', 'image2text']:
            pred_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return dict(
                gt_pose=body_poseA_rotmat.to(torch.float32), 
                pred_text=pred_text, 
                gt_text=caption)
        elif self.evaluate_task in ['text2pose', 'pose_edit']:
            pred_body_pose_rotmat = self.decode_pose_from_outputs(outputs, self.device, input_ids.dtype)
            return dict(
                pred_pose=pred_body_pose_rotmat.to(torch.float32), 
                gt_pose=body_poseA_rotmat.to(torch.float32), 
                gt_text=caption)
        elif self.evaluate_task in ['image2pose']:
            pred_body_pose = self.decode_pose_from_outputs(
                outputs, self.device, input_ids.dtype, return_pose_type='axis_angle')
            
            return dict(
                # for debug, vis generated mesh
                # gt_body_pose=gt_body_pose, 
                # pred_body_pose=pred_body_pose, 
                
                pred_axis_angles=pred_body_pose, 
                keypoints_3d=keypoints_3d)
        elif self.evaluate_task in ['pose_difference', 'image_difference']:
            pred_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gt_pose = torch.stack([body_poseA_rotmat.to(torch.float32), 
                                   body_poseB_rotmat.to(torch.float32)], dim=0)
            return dict(
                gt_pose=gt_pose, 
                pred_text=pred_text, 
                gt_text=caption
            )
        else:
            raise NotImplementedError
    
    def decode_pose_from_outputs(self, output_ids, device, dtype, return_pose_type='rotmat'):
        '''
        generate pose id from model.generate() outputs, then decode pose id using pose_vqvae decode
        '''
        pose_tokens = torch.zeros((output_ids.shape[0], self.model.pose_vqvae.token_num), dtype=dtype).to(device)
        # extract pose tokens
        for i, output in enumerate(output_ids):
            begin_idx = torch.where(output == self.pose_begin_idx)[0][0]
            end_idx = torch.where(output == self.pose_end_idx)[0][0]
            output = output[begin_idx + 1:end_idx]
            min_len = min(pose_tokens.shape[1], output.shape[0])
            pose_tokens[i, :min_len] = output[:min_len]
        pose_strings = self.tokenizer.batch_decode(pose_tokens)
        pose_ids = torch.tensor([
            [int(item.split('_')[-1].replace('>', '')) 
                for item in pose_string.split('> <')] 
            for pose_string in pose_strings], dtype=dtype).to(device)
        pred_body_pose_rotmat = self.model.pose_vqvae.decode(pose_ids, return_type=return_pose_type)
        return pred_body_pose_rotmat

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, hmr_images=None, tasks=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features = self.get_model().get_vision_tower()(images) # bs, 24*24, 1024
        
        # tokenhmr vit backbone image features
        hmr_image_features = self.model.hmr_vit_backbone(hmr_images[..., 32:-32]) # bs, 3, 256, 192 -> bs, 1280, 16, 12
        hmr_image_features = F.pad(hmr_image_features, (2, 2), mode='constant', value=0)
        hmr_image_features = F.interpolate(hmr_image_features, size=(24, 24), mode='bilinear', align_corners=True)
        hmr_image_features = hmr_image_features.flatten(2).transpose(1, 2) # bs, 24*24, 1280
        image_features = torch.cat([image_features, hmr_image_features], dim=-1)
        image_features = self.get_model().mm_projector(image_features)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) if position_ids is None else position_ids
        labels = torch.full_like(input_ids, IGNORE_INDEX) if labels is None else labels

        # remove the padding using attention_mask -- FIXME
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            assert num_images.item() in [0, 1, 2]
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 2
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            if num_images == 1:
                cur_image_idx += 1

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    @torch.no_grad()
    def generate(self, inputs, images, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        hmr_images = kwargs.pop('hmr_images', None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            inputs, position_ids, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, hmr_images)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        outputs = MistralModel.generate(self, position_ids=position_ids, attention_mask=attention_mask, 
                                        inputs_embeds=inputs_embeds, **kwargs)
        return outputs
    

class PoseGPTCLIP(PoseGPT):
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, hmr_images=None, tasks=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features = self.get_model().get_vision_tower()(images) # bs, 24*24, 1024
        image_features = self.get_model().mm_projector(image_features)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) if position_ids is None else position_ids
        labels = torch.full_like(input_ids, IGNORE_INDEX) if labels is None else labels

        # remove the padding using attention_mask -- FIXME
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            assert num_images.item() in [0, 1, 2]
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 2
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            if num_images == 1:
                cur_image_idx += 1

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

# ===============================================================
# posegpt utils
# ===============================================================

def process_templates(captions, tasks, poseA_tokens, poseB_tokens, 
                      tokenizer=None, codebook_size=2048):
    poseA_strings = batch_pose_tokenids2string(poseA_tokens, codebook_size=codebook_size)
    poseB_strings = batch_pose_tokenids2string(poseB_tokens, codebook_size=codebook_size)
    
    # process batch template
    input_ids, output_ids = [], []
    conv = conversation_lib.default_conversation.copy()

    for i in range(len(captions)):
        conv.messages = []
        input_prompt = process_template(tasks[i]['input'], captions[i], poseA_strings[i], poseB_strings[i])
        conv.append_message(conv.roles[0], input_prompt)
        inputs_len = len(tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors='pt'))
        conv.append_message(conv.roles[1], None)
        input_id = tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors='pt')
        input_ids.append(input_id)

    # batch inference need left padding
    reversed_seq = [seq.flip(0) for seq in input_ids]
    padded_seq = torch.nn.utils.rnn.pad_sequence(
        reversed_seq, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_seq = padded_seq.flip(1)

    attention_mask = padded_seq.ne(tokenizer.pad_token_id)
    input_ids = padded_seq[:, -tokenizer.model_max_length:]
    return input_ids, attention_mask

# process single template
def process_template(template, caption, poseA_string, poseB_string):
    # process pose and caption
    prompt = template.replace(CAPTION_TOKEN, caption)
    # process pose
    prompt = prompt.replace(POSE_TOKEN, POSEA_TOKEN)\
                    .replace(POSEA_TOKEN, poseA_string)\
                    .replace(POSEB_TOKEN, poseB_string)\
    # process image
    prompt = prompt.replace(IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    # multi image
    return prompt

def batch_pose_tokenids2string(batch_tokenids: torch.Tensor, codebook_size=2048):
    '''
    Args:
        tokenids: [bs, n]
    Example:
        [0, 2, 9] -> <pose_start><pose_id_0><pose_id_2><pose_id_9><pose_end>
    '''
    assert len(batch_tokenids.shape) == 2
    strings = []
    for tokenids in batch_tokenids:
        mid_string = ''.join([f'<pose_id_{i}>' for i in tokenids])
        strings.append(f'<pose_id_{codebook_size}>' + mid_string + f'<pose_id_{codebook_size + 1}>')
    return strings

def replace_pose_token_with_incorrect_token(pose_tokens, mask_ratio=0.15, num_tokens=2048):
    _pose_tokens = pose_tokens.clone()
    mask = torch.bernoulli(torch.zeros_like(pose_tokens), p=mask_ratio)
    rand_id = torch.randint_like(_pose_tokens, high=num_tokens)
    _pose_tokens = torch.where(mask.to(torch.bool), rand_id, _pose_tokens)
    return _pose_tokens

def get_pose_query_token_string(num_tokens=80, codebook_size=2048):
    return (f'<pose_id_{codebook_size}>' + 
            ''.join([f'<pose_query_{i}>' for i in range(num_tokens)]) + 
            f'<pose_id_{codebook_size + 1}>')
