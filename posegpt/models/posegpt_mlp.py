import torch
from posegpt.constants import *
from posegpt.utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
import torch.nn.functional as F
from transformers import MistralModel
from transformers.models.mistral.modeling_mistral import add_start_docstrings_to_model_forward, MISTRAL_INPUTS_DOCSTRING, logger, DynamicCache, BaseModelOutputWithPast, Cache, CausalLMOutputWithPast, CrossEntropyLoss, _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask
from typing import * 
from .posegpt import process_templates, PoseGPTModel, PoseGPT
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerateNonBeamOutput, dist, ModelOutput
import torch.nn as nn
from posegpt.utils.vis_utils import vis_mesh
from ..utils import BodyModel

class PoseGPTMLP(PoseGPT):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = PoseGPTModel(config)

        self.smpl_decoder = torch.nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 22 * 6),
            nn.Dropout(0.0), 
        )
        self.body_model = BodyModel('processed_dataset/smpl_models/smplh/SMPLH_NEUTRAL.npz', dtype=torch.bfloat16)
        self.joint_loss = nn.MSELoss()

    def forward(self, 
                # original input, for inference
                input_ids=None, attention_mask=None, position_ids=None, 
                past_key_values=None, inputs_embeds=None, labels=None, 

                # from dataset, for training
                body_poseA_rotmat=None, body_poseB_rotmat=None, images=None, 
                caption=None, tasks=None, 
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs):
        if self.training:
            # pose tokenizer
            poseA_tokens = torch.zeros((body_poseA_rotmat.shape[0], 1)).to(body_poseA_rotmat.device).to(torch.int64)
            poseB_tokens = torch.zeros_like(poseA_tokens)

            input_ids, output_ids, attention_mask = process_templates( # type: ignore
                caption, tasks, poseA_tokens, poseB_tokens, tokenizer=self.tokenizer, 
                codebook_size=self.pose_vqvae_codebook_size) # type: ignore
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, 
            labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids.to(self.device), position_ids=position_ids, attention_mask=attention_mask.to(self.device), 
                past_key_values=past_key_values, labels=output_ids.to(self.device), images=images, tasks=tasks, 
                hmr_images=kwargs.get('hmr_images', None))

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            use_cache=use_cache, 
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # hmr loss
            hmr_loss = 0
            for i, label in enumerate(labels):
                pose_token_id = torch.where(label == (self.pose_begin_idx - 1))[0]
                rot_6d = self.smpl_decoder(hidden_states[i][pose_token_id])
                rotmat = rotation_6d_to_matrix(rot_6d.to(torch.float32).view(-1, 22, 6)).view(-1, 22, 3, 3) # 1, 22, 3, 3
                pred_axis_angle = matrix_to_axis_angle(rotmat).flatten(1, 2).to(torch.bfloat16) # 1, 22, 3 -> 1, 66
                gt_axis_angle = matrix_to_axis_angle(body_poseA_rotmat[i])[None].flatten(1, 2).to(torch.bfloat16) # (1, 22, 3) -> 1, 66

                pred_joints = self.body_model(root_orient=pred_axis_angle[:, :3], pose_body=pred_axis_angle[:, 3:]).Jtr
                gt_joints = self.body_model(root_orient=gt_axis_angle[:, :3], pose_body=gt_axis_angle[:, 3:]).Jtr

                theta_loss = (body_poseA_rotmat[i:i+1] - rotmat).abs().mean() * 0.01
                joint_loss = self.joint_loss(pred_joints, gt_joints) * 0.2
                
                hmr_loss = hmr_loss + theta_loss + joint_loss
            
            loss = loss + hmr_loss / labels.shape[0]

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def evaluate(self, body_poseA_rotmat, body_poseB_rotmat, images, caption, tasks, **kwargs):
        # pose tokenizer
        keypoints_3d = kwargs.pop('keypoints_3d', None)
        poseA_tokens = torch.zeros((body_poseA_rotmat.shape[0], 1)).to(body_poseA_rotmat.device).to(torch.int64)
        poseB_tokens = torch.zeros_like(poseA_tokens)
        input_ids, attention_mask = process_templates(
            caption, tasks, poseA_tokens, poseB_tokens, tokenizer=self.tokenizer, 
            training=False, codebook_size=self.pose_vqvae_codebook_size)
        # generate
        self.config.tokenizer_padding_side = 'left'
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        outputs = self.generate(
            input_ids, attention_mask=attention_mask, images=images, max_new_tokens=512, 
            output_hidden_states=True, return_dict_in_generate=True, use_cache=True, 
            num_beams=1, **kwargs)
        # convert output to float32 for metric calculation.
        # pose process
        if self.evaluate_task in ['image2pose']:
            # 只能说离大谱了
            # from posegpt.utils.vis_utils import vis_mesh
            # rot6d = self.smpl_decoder(outputs.hidden_states[3][-1]).view(body_poseA_rotmat.shape[0], 22, 6)
            # rotmat = rotation_6d_to_matrix(rot6d.to(torch.float32).cpu())
            # rotmat[:, 8, 0, 0] = -rotmat[:, 8, 0, 0];rotmat[:, 8, 1, 1] = -rotmat[:, 8, 1, 1];rotmat[:, 4, 0, 0] = -rotmat[:, 4, 0, 0]; rotmat[:, 4, 1, 2] = -rotmat[:, 4, 1, 2]; rotmat[:, 4, 2, 2] = -rotmat[:, 4, 2, 2]; rotmat[:, 15, 0, 0] = -rotmat[:, 15, 0, 0]; rotmat[:, 15, 1, 1] = -rotmat[:, 15, 1, 1]
            # axis_angle = matrix_to_axis_angle(rotmat);vis_mesh(axis_angle[0].flatten().cpu())
            # axis_angle = matrix_to_axis_angle(body_poseA_rotmat);vis_mesh(axis_angle[0].flatten().cpu())
            rot6d = self.smpl_decoder(outputs.hidden_states[3][-1]).view(body_poseA_rotmat.shape[0], 22, 6)
            rotmat = rotation_6d_to_matrix(rot6d.to(torch.float32).cpu())
            rotmat[:, 8, 0, 0] = -rotmat[:, 8, 0, 0];rotmat[:, 8, 1, 1] = -rotmat[:, 8, 1, 1];rotmat[:, 4, 0, 0] = -rotmat[:, 4, 0, 0]; rotmat[:, 4, 1, 2] = -rotmat[:, 4, 1, 2]; rotmat[:, 4, 2, 2] = -rotmat[:, 4, 2, 2]; rotmat[:, 15, 0, 0] = -rotmat[:, 15, 0, 0]; # rotmat[:, 15, 1, 1] = -rotmat[:, 15, 1, 1]
            axis_angle = matrix_to_axis_angle(rotmat)
            return dict(
                pred_axis_angles=axis_angle, 
                keypoints_3d=keypoints_3d)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def generate(self, inputs, images, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        hmr_images = kwargs.pop('hmr_images', None)
        tasks = kwargs.pop('tasks', None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            inputs, position_ids, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, hmr_images, tasks=tasks)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        outputs = MistralModel.generate(self, position_ids=position_ids, attention_mask=attention_mask, 
                                        inputs_embeds=inputs_embeds, **kwargs)
        return outputs
