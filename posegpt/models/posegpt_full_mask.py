import torch
from posegpt.constants import *
from posegpt.utils.rotation_conversions import matrix_to_axis_angle
import torch.nn.functional as F
from transformers import MistralModel
from transformers.models.mistral.modeling_mistral import add_start_docstrings_to_model_forward, MISTRAL_INPUTS_DOCSTRING, logger, DynamicCache, BaseModelOutputWithPast, Cache, CausalLMOutputWithPast, CrossEntropyLoss, _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask
from typing import * 
from .posegpt import process_templates, PoseGPTModel, PoseGPT
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerateNonBeamOutput, dist, ModelOutput


class PoseGPTModelFullMask(PoseGPTModel):
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None, # type: ignore
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...") # type: ignore
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache: past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device) # type: ignore
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length) # type: ignore
        else:
            position_ids = position_ids.view(-1, seq_length).long() # type: ignore

        if inputs_embeds is None:
            if torch.any(input_ids == -1): # 必须是pose begin idx + [-1, ] * 80, replace -1 to pose embed
                assert input_ids.shape[-1] == 81
                inputs_embeds = torch.concat([self.embed_tokens(input_ids[:, :1]), 
                                              self.embed_tokens.weight[None, -80:, :].repeat(input_ids.shape[0], 1, 1)], dim=1) # type: ignore
            else:
                inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. ")
            
        # TODO: add comment
        if self.training or seq_length == 1 or seq_length == attention_mask.shape[-1]:
            causal_attn_mask = _prepare_4d_causal_attention_mask_for_sdpa( # bs, 1, seq_length, all_input_id_length
                attention_mask.to(torch.bool), (batch_size, seq_length), inputs_embeds, past_key_values_length)
            if causal_attn_mask is None: # TODO: why?
                causal_attn_mask = _prepare_4d_causal_attention_mask( # bs, 1, seq_length, all_input_id_length
                    attention_mask.to(torch.bool), (batch_size, seq_length), inputs_embeds, past_key_values_length)
        else: # seq_length == 81
            causal_attn_mask = _prepare_4d_causal_attention_mask_for_sdpa(  # bs, 1, all_input_id_length, all_input_id_length
                attention_mask.to(torch.bool), attention_mask.shape, inputs_embeds, 0)
            if causal_attn_mask is None:
                causal_attn_mask = _prepare_4d_causal_attention_mask( # bs, 1, seq_length, all_input_id_length
                    attention_mask.to(torch.bool), (batch_size, seq_length), inputs_embeds, past_key_values_length)
            else:
                causal_attn_mask = causal_attn_mask[:, :, -seq_length:]

        for i in range(attention_mask.shape[0]):
            full_mask_idx = torch.where(attention_mask[i] == 2)[0]
            if full_mask_idx.size(0) > 0:
                _min = full_mask_idx.min()
                _max = full_mask_idx.max()
                ignore_cnt = causal_attn_mask.shape[-1] - causal_attn_mask.shape[-2]
                causal_attn_mask[i, 0, _min - ignore_cnt:_max + 1 - ignore_cnt, _min:_max + 1] = 0
        
        attention_mask = causal_attn_mask

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__, hidden_states, attention_mask,
                    position_ids, past_key_values, output_attentions, use_cache,)
            else:
                layer_outputs = decoder_layer(
                    hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_value=past_key_values, output_attentions=output_attentions, use_cache=use_cache,)

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],) # type: ignore

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,) # type: ignore

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache # type: ignore

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states, # type: ignore
            attentions=all_self_attns) # type: ignore


class PoseGPTFullMask(PoseGPT):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = PoseGPTModelFullMask(config)

    def forward(
        self,
        # from dataset, for training
        body_poseA_rotmat=None, body_poseB_rotmat=None, images=None, caption=None, tasks=None, 

        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

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
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def evaluate(self, body_poseA_rotmat, body_poseB_rotmat, images, caption, tasks, **kwargs):
        poseA_tokens = self.model.pose_vqvae.encode(body_poseA_rotmat)
        poseB_tokens = self.model.pose_vqvae.encode(body_poseB_rotmat)
        input_ids, attention_mask = process_templates(
            caption, tasks, poseA_tokens, poseB_tokens, tokenizer=self.tokenizer, 
            codebook_size=self.pose_vqvae_codebook_size)
        self.config.tokenizer_padding_side = 'left'
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)

        outputs = self.generate(
            input_ids, attention_mask=attention_mask, images=images, max_new_tokens=512, 
            num_beams=1, use_cache=True, tasks=tasks, **kwargs)

        pred_body_pose = None
        pred_text = None
        if torch.where(outputs == self.pose_begin_idx)[0].shape[0] > 0:
            pred_body_pose = self.decode_pose_from_outputs(
                outputs, self.device, input_ids.dtype, return_pose_type='axis_angle')
        else:
            pred_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return dict(
            body_pose=pred_body_pose, 
            text=pred_text)

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

        new_input_embeds, new_labels, cur_image_idx = [], [], 0
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
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
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
        
        attention_mask = attention_mask.to(torch.int8)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

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
    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer=None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        assert streamer is None
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            raise NotImplementedError
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        eos_token_id = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        
        return_dict_in_generate = (return_dict_in_generate if return_dict_in_generate is not None 
                                   else self.generation_config.return_dict_in_generate)
        if return_dict_in_generate: raise NotImplementedError

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)

        # used by synced_gpus only
        this_peer_finished, next_pose_prediction = False, False
        while True:
            if synced_gpus:
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                if this_peer_finished_flag.item() == 0.0: break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True, output_attentions=output_attentions,
                           output_hidden_states=output_hidden_states)
            # don't waste resources running the code we don't need
            if synced_gpus and this_peer_finished: continue  
            
            if next_pose_prediction:
                next_token_logits = outputs.logits[:, -80:, :]
                next_pose_prediction = False
            else:
                next_token_logits = outputs.logits[:, -1:, :]
            
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1) 
            # next pose prediction
            if torch.any(next_tokens == self.pose_begin_idx):
                next_tokens[:] = self.pose_begin_idx
                # assert torch.all(next_tokens == self.pose_begin_idx)
                next_pose_prediction = True

            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, 
                is_encoder_decoder=self.config.is_encoder_decoder, next_pose_prediction=next_pose_prediction)
            
            if next_pose_prediction:
                next_tokens = torch.concat([next_tokens, -torch.ones_like(next_tokens).repeat(1, 80)], dim=1)
            else:
                # finished sentences should have their next token be a padding token
                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    unfinished_sequences = unfinished_sequences[:, -next_tokens.shape[-1]:]
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            if next_tokens.shape[-1] == 80:
                input_ids[:, -80:] = next_tokens
                pose_end_token = torch.zeros_like(next_tokens[:, :1])
                pose_end_token[:] = self.pose_end_idx
                input_ids = torch.cat([input_ids, pose_end_token], dim=-1)
            else:
                input_ids = torch.cat([input_ids, next_tokens], dim=-1) # type: ignore

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
                # stop when each sentence is finished
                this_peer_finished = True if unfinished_sequences.max() == 0 else this_peer_finished

            # stop if we exceed the maximum length
            this_peer_finished = True if stopping_criteria(input_ids, scores) else this_peer_finished
            if this_peer_finished and not synced_gpus: break

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        next_pose_prediction: bool = False, 
    ) -> Dict[str, Any]:
        assert not is_encoder_decoder 
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format)
        if getattr(outputs, "state", None) is not None: raise NotImplementedError
        if "token_type_ids" in model_kwargs: raise NotImplementedError

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            if next_pose_prediction:
                attention_mask = torch.cat([attention_mask, 2*attention_mask.new_ones((attention_mask.shape[0], 80))], dim=-1)
            model_kwargs['attention_mask'] = attention_mask
        return model_kwargs

    def prepare_inputs_for_generation( # type: ignore
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                raise NotImplementedError
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.bool().long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask})
        return model_inputs

class PoseGPTFullMaskOnlyPoseVit(PoseGPTFullMask):
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, hmr_images=None, tasks=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # image_features = self.get_model().get_vision_tower()(images) # bs, 24*24, 1024
        # tokenhmr vit backbone image features
        hmr_image_features = self.model.hmr_vit_backbone(hmr_images[..., 32:-32]) # bs, 3, 256, 192 -> bs, 1280, 16, 12
        hmr_image_features = F.pad(hmr_image_features, (2, 2), mode='constant', value=0)
        hmr_image_features = F.interpolate(hmr_image_features, size=(24, 24), mode='bilinear', align_corners=True)
        hmr_image_features = hmr_image_features.flatten(2).transpose(1, 2) # bs, 24*24, 1280
        # image_features = torch.cat([image_features, hmr_image_features], dim=-1)
        image_features = self.get_model().mm_projector(hmr_image_features)

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

        new_input_embeds, new_labels, cur_image_idx = [], [], 0
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
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
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
        
        attention_mask = attention_mask.to(torch.int8)
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
