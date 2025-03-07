import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from posegpt.models.base_module import BaseModule, build_model
import numpy as np
from transformers import AutoTokenizer
import os.path as osp
from posegpt.utils.rotation_conversions import matrix_to_axis_angle
from .components.resnet import Resnet1D
from posegpt.utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix

class TextPoseRetrieval(BaseModule):
    '''
    text-pose retrieval
    '''
    def __init__(self, pose_encoder, text_encoder, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pose_encoder = build_model(pose_encoder)
        self.text_encoder = build_model(text_encoder)  

    def forward(self, body_pose_rotmat, caption):
        text_embd = self.text_encoder(caption)
        pose_embd = self.pose_encoder(body_pose_rotmat)
        return pose_embd, text_embd

    def training_step(self, batch, batch_idx):
        pose_embd, text_embd = self.forward(batch['body_pose_rotmat'], batch['caption'])
        loss = self.loss.update(pose_embd, text_embd)
        self.log('loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pose_embd, text_embd = self.forward(batch['body_pose_rotmat'], batch['caption'])
        self.metric.update(
             pred_pose_embd=pose_embd, 
             pred_text_embd=text_embd)

class TextPoseABRetrieval(TextPoseRetrieval):
    '''
    text-poseAB retrieval, for posefix dataset
    '''
    def forward(self, body_poseA_rotmat, body_poseB_rotmat, caption):
        text_embd = self.text_encoder(caption)
        pose_embd = self.pose_encoder(body_poseA_rotmat, body_poseB_rotmat)
        return pose_embd, text_embd

    def training_step(self, batch, batch_idx):
        pose_embd, text_embd = self.forward(
            batch['body_poseA_rotmat'], batch['body_poseB_rotmat'], batch['caption'])
        loss = self.loss.update(pose_embd, text_embd)
        self.log('loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pose_embd, text_embd = self.forward(
            batch['body_poseA_rotmat'], batch['body_poseB_rotmat'], batch['caption'])
        self.metric.update(
             pred_pose_embd=pose_embd, 
             pred_text_embd=text_embd)

################################################################################
## Text encoders
################################################################################

class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(TextEncoderBiGRUCo, self).__init__()

        # 用distill bert里面的word embd
        llm_base = 'cache/distilbert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(llm_base, legacy=True)
        word_embd = torch.load(osp.join(llm_base, 'pytorch_model.bin'))[
             'distilbert.embeddings.word_embeddings.weight']
        # freeze word embd has better performance
        self.word_embd = nn.Embedding(word_embd.shape[0], word_embd.shape[1], _weight=word_embd, _freeze=True).cpu()
        self.max_length = 196

        self.output_size = output_dim
        self.input_emb = nn.Sequential(
             nn.Linear(word_embd.shape[1], hidden_dim), 
             nn.LayerNorm(hidden_dim),
             nn.Linear(hidden_dim, hidden_dim))
        
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        output_net = [
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            L2Norm(), 
        ]

        self.output_net = nn.Sequential(*output_net)

        self.hidden_size = hidden_dim
        self.hidden = nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=True)
        )

    def forward(self, caption):
        tokens = self.tokenizer(caption, max_length=self.max_length, padding='max_length', 
                                truncation=True, return_tensors='pt')
        device = self.word_embd.weight.device
        token_ids = tokens.input_ids.to(device)
        cap_lens = tokens.attention_mask.sum(1)

        word_embs = self.word_embd(token_ids) # bs, n, c
        num_samples = token_ids.shape[0]

        # 需要按照升序排列
        # initialize output
        out = torch.zeros((num_samples, self.output_size), dtype=torch.float32, device=device)
        # provide data to the model in decreasing length order
        asort = torch.argsort(-cap_lens)
        word_embs = word_embs[asort,:]
        cap_lens = cap_lens[asort]
        input_embs = self.input_emb(word_embs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input=input_embs, lengths=cap_lens, batch_first=True)
        _, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        # reorder data as it was before
        _out = self.output_net(gru_last)
        out[asort, :] = _out
        return out

################################################################################
## poseencoder
################################################################################

class VPoserPoseEncoder(nn.Module):
    '''
    copy from posescript
    '''
    def __init__(self, input_dim=66, hidden_dim=512, hidden_dim_mini=32, output_dim=512, **kwargs):
        super(VPoserPoseEncoder, self).__init__()
        
        # use VPoser pose encoder architecture...
        encoder_layers = [
            BatchFlatten(),
            nn.BatchNorm1d(input_dim), 
            nn.Linear(input_dim, hidden_dim), 
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ]
        
        # output layers
        encoder_layers += [
            # keep the bottleneck while adapting to the joint embedding size
            nn.Linear(hidden_dim, hidden_dim_mini), 
            nn.ReLU(),
            nn.Linear(hidden_dim_mini, output_dim),
            L2Norm(),
        ]

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, body_pose, pose_type='rotmat'):
        if pose_type == 'rotmat':
            body_pose_axis_angle = matrix_to_axis_angle(body_pose).flatten(1, 2)
        elif pose_type == 'axis_angle':
            body_pose_axis_angle = body_pose
        elif pose_type == 'rot6d':
            import ipdb; ipdb.set_trace()
            body_pose_rotmat = rotation_6d_to_matrix(body_pose)
            body_pose_axis_angle = matrix_to_axis_angle(body_pose_rotmat).flatten(1, 2)
        else:
            raise NotImplementedError
        return self.encoder(body_pose_axis_angle)

class VPoserPoseABEncoder(VPoserPoseEncoder):
    '''
    copy from posescript
    '''
    def __init__(self, output_dim=512, **kwargs):
        super(VPoserPoseABEncoder, self).__init__(output_dim=output_dim, **kwargs)
        
        # merge
        self.pose_mlp = nn.Sequential(
			ConCatModule(),
			nn.Linear(2 * output_dim, 2 * output_dim),
			nn.LeakyReLU(),
			nn.Linear(2 * output_dim, output_dim),
			nn.LeakyReLU(),
			nn.Linear(output_dim, output_dim),
			nn.LeakyReLU(),
			L2Norm()
        )

    def forward(self, body_poseA_rotmat, body_poseB_rotmat):
        body_poseA_axis_angle = matrix_to_axis_angle(body_poseA_rotmat).flatten(1, 2)
        body_poseB_axis_angle = matrix_to_axis_angle(body_poseB_rotmat).flatten(1, 2)
        xA = self.encoder(body_poseA_axis_angle)
        xB = self.encoder(body_poseB_axis_angle)
        x = self.pose_mlp([xA, xB])
        return x

class VQPoseEncoder(nn.Module):
    '''
    copy from pose VQVAE, add 1D conv to reduce dim. 
    '''
    def __init__(self,
                 input_dim=6,
                 hidden_dim=512, 
                 output_dim=512, 
                 upsample_step=1, 
                 downsample_step=1,
                 num_joints=22, 
                 **kwargs):
        super(VQPoseEncoder, self).__init__()
        encoder_layers = []
        
        encoder_layers.append(nn.Conv1d(input_dim, hidden_dim, 3, 1, 1))
        encoder_layers.append(nn.ReLU())

        # Make num of tokens in multiple of 10
        up_sample_tokens = ((num_joints * 2) // 10) * 10
        encoder_layers.append(nn.Upsample(up_sample_tokens)) # 40 token
        encoder_layers.append(nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1))
        encoder_layers.append(nn.ReLU())

        for _ in range(upsample_step): # 40 -> 80
            # token num upsample 2 times
            encoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            encoder_layers.append(nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1))
            encoder_layers.append(nn.ReLU())

        # token num downsample 2 times
        for _ in range(downsample_step): # 80 -> 40
            encoder_layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(
                Resnet1D(hidden_dim, n_depth=2, dilation_growth_rate=3, activation='relu', norm=False))
        encoder_layers.append(nn.Conv1d(hidden_dim, output_dim, 3, 1, 1))
        self.encoder = nn.Sequential(*encoder_layers)

        self.output_layer = nn.Sequential(
             nn.Conv1d(up_sample_tokens * upsample_step // downsample_step, 1, 3, 1, 1), 
             L2Norm(),)

    def forward(self, body_pose, pose_type='rotmat'):
        if pose_type == 'rotmat':
            body_pose_rot6d = matrix_to_rotation_6d(body_pose)
        elif pose_type == 'rot6d':
            body_pose_rot6d = body_pose
        elif pose_type == 'axis_angle':
            body_pose_rotmat = axis_angle_to_matrix(body_pose)
            body_pose_rot6d = matrix_to_rotation_6d(body_pose_rotmat)
        else:
            raise NotImplementedError

        # (bs, num_joints, 6) -> (bs, 6, num_joints)
        x = body_pose_rot6d.permute(0, 2, 1)
        # bs, c, token_num -> bs, token_num, c
        x = self.encoder(x).permute(0, 2, 1)
        # bs, token_num, c -> bs, c
        x = self.output_layer(x).squeeze(1)
        return x

class VQPoseABEncoder(VQPoseEncoder):
    '''
    copy from pose VQVAE, add 1D conv to reduce dim, add MLP to merge feature of pose A & B
    '''
    def __init__(self, output_dim=512, **kwargs):
        super(VQPoseABEncoder, self).__init__(output_dim=output_dim, **kwargs)
        self.merge_later = nn.Sequential(
            nn.Conv1d(output_dim * 2, output_dim * 2, 3, 1, 1), 
            nn.ReLU(), 
            nn.Conv1d(output_dim * 2, output_dim, 3, 1, 1), 
            nn.ReLU(), 
            nn.Conv1d(output_dim, output_dim, 3, 1, 1))

    def forward(self, body_poseA_rotmat, body_poseB_rotmat):
        # (bs, num_joints, 6) -> (bs, 6, num_joints)
        xA = matrix_to_rotation_6d(body_poseA_rotmat).permute(0, 2, 1)
        xB = matrix_to_rotation_6d(body_poseB_rotmat).permute(0, 2, 1)
        # bs, c, token_num -> bs, token_num, c
        xA = self.encoder(xA).permute(0, 2, 1)
        xB = self.encoder(xB).permute(0, 2, 1)
        # merge 
        # bs, token_num, c -> bs, 2c, token_num
        x = torch.cat([xA, xB], dim=-1).permute(0, 2, 1)
        # bs, 2c, token_num -> bs, token_num, c
        x = self.merge_later(x).permute(0, 2, 1)
        # bs, token_num, c -> bs, c
        x = self.output_layer(x).squeeze(1)

        return x

################################################################################
## modules
################################################################################

class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = 'batch_flatten'

    def forward(self, x):
        return x.view(x.shape[0], -1)

class L2Norm(nn.Module):
	def forward(self, x):
		return x / x.norm(dim=-1, keepdim=True)

class ConCatModule(nn.Module):

	def __init__(self):
		super(ConCatModule, self).__init__()

	def forward(self, x):
		x = torch.cat(x, dim=1)
		return x
     