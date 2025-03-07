import torch
import torch.nn as nn
from typing import Dict
from posegpt.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from posegpt.utils import BodyModel
import numpy as np
from .components.resnet import Resnet1D
from torch import Tensor
from .base_module import build_model, BaseModule

class PoseVQVAE(BaseModule):
    def __init__(self, 
                 encoder: Dict=None, 
                 decoder: Dict=None, 
                 quantizer: Dict=None, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = build_model(encoder)
        self.decoder = build_model(decoder)
        self.quantizer = build_model(quantizer)

        self.body_model = BodyModel('processed_dataset/smpl_models/smplh/SMPLH_NEUTRAL.npz')
        self.codebook_size = self.quantizer.nb_code
        self.token_num = int(self.decoder.token_num)
    
    def forward(self, batch):
        '''
        batch: 
            body_joints: b,J,3
            body_vertices: b,n
            body_pose_axis_angle: b,J,3
            body_pose_rotmat: b,J,3,3
        kwargs: 
            current_epoch: 当前epoch
        '''
        # rotmat -> 6d mat
        body_pose_6d = matrix_to_rotation_6d(batch['body_pose_rotmat']) # bs, 21, 6

        # Encode
        x_encoder = self.encoder(body_pose_6d, self.trainer.global_step)

        # quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        # x_quantized: bs, c, n

        # decoder
        pred_body_pose_6d = self.decoder(x_quantized)
        pred_body_pose_rotmat = rotation_6d_to_matrix(pred_body_pose_6d)
        pred_body_pose_axis_angle = matrix_to_axis_angle(pred_body_pose_rotmat).flatten(1, 2)
        body_mesh = self.body_model(
            root_orient=pred_body_pose_axis_angle[:, :3], 
            pose_body=pred_body_pose_axis_angle[:, 3:])
        
        output = dict(
            pred_body_pose_rotmat=pred_body_pose_rotmat, 
            pred_body_pose_axis_angle=pred_body_pose_axis_angle, 
            pred_body_vertices=body_mesh.v, 
            pred_body_joints=body_mesh.Jtr, 
            loss_commit=loss)
        return output
    
    # to fetch code idx
    def encode(self, body_pose_rotmat):
        # rotmat -> 6d mat
        body_pose_6d = matrix_to_rotation_6d(body_pose_rotmat)
        bs = body_pose_6d.shape[0]
        x_encoder = self.encoder(body_pose_6d) # bs, C, num_token
        x_encoder = self.quantizer.preprocess(x_encoder) # bs*num_token, C
        code_idx = self.quantizer.quantize(x_encoder) # bs * num_token
        code_idx = code_idx.view(bs, -1)

        # latent, dist
        return code_idx
    
    def decode(self, x: Tensor, return_type='rotmat'):
        # x为code_idx, shape=[bs, num_tokens]
        x_d = self.quantizer.dequantize(x) # bs, num_tokens, c
        x_d = x_d.permute(0, 2, 1).contiguous() # bs, c, num_tokens

        # decoder
        pred_body_pose_6d = self.decoder(x_d) # bs, num_joints, out_dim
        
        # ============vis===============
        # from processed_dataset.vis_utils import vis_mesh
        # pred_body_pose_rotmat = rotation_6d_to_matrix(pred_body_pose_6d.to(torch.float32))
        # pred_body_pose_axis_angle = matrix_to_axis_angle(pred_body_pose_rotmat.to(torch.float32))
        # vis_mesh(pred_body_pose_axis_angle.flatten().cpu())
        # ==============================

        # NOTE: RuntimeError: "cross_cuda" not implemented for 'BFloat16'
        pred_body_pose_rotmat = rotation_6d_to_matrix(pred_body_pose_6d.to(torch.float32))
        pred_body_pose_rotmat = pred_body_pose_rotmat.to(pred_body_pose_6d)

        if return_type == 'rotmat':
            return pred_body_pose_rotmat
        elif return_type == 'axis_angle':
            return matrix_to_axis_angle(pred_body_pose_rotmat).flatten(1, 2)
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # pose encode & decode
        output = self.forward(batch)
        loss = self.loss.update(pred_dicts=output, gt_dicts=batch)
        self.log('loss', loss.detach(), logger=True, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # pose encode & decode
        output = self.forward(batch)
        self.metric.update(output["pred_body_pose_axis_angle"], batch["body_pose_axis_angle"])

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure=None,
    ) -> None:
        if self.trainer.global_step < 200:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.optimizer_config['params']['lr']
        optimizer.step(closure=optimizer_closure)

# --------------------------------------------------------
# copy from tokenhmr
# https://github.com/saidwivedi/TokenHMR/blob/e7288b65b83895261aaa64a7ee29d5cc3876eb53/tokenization/models/vanilla_pose_vqvae.py
# --------------------------------------------------------

class TokenHMREncoder(nn.Module):
    def __init__(self,
                 input_dim=6,
                 hidden_dim=512, 
                 output_dim=512, 
                 upsample_step=1, 
                 downsample_step=1,
                 num_joints=21):
        super(TokenHMREncoder, self).__init__()
        encoder_layers = []
        
        # 升维
        encoder_layers.append(nn.Conv1d(input_dim, hidden_dim, 3, 1, 1))
        encoder_layers.append(nn.ReLU())

        # Make num of tokens in multiple of 10
        encoder_layers.append(nn.Upsample(((num_joints*2)//10)*10)) # 40 token
        encoder_layers.append(nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1))
        encoder_layers.append(nn.ReLU())

        for _ in range(upsample_step): # 40 -> 160
            # token num 上采样2倍
            encoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            encoder_layers.append(nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1))
            encoder_layers.append(nn.ReLU())

        # token num 下采样2倍
        for _ in range(downsample_step): # 160 -> 80
            encoder_layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(
                Resnet1D(hidden_dim, n_depth=2, dilation_growth_rate=3, activation='relu', norm=False))

        encoder_layers.append(nn.Conv1d(hidden_dim, output_dim, 3, 1, 1))
        self.encoder = nn.Sequential(*encoder_layers)

        self.epoch_noise_scale_mapping = {
            0: 1e-2, 1: 5e-2, 2: 1e-1, 3: 1e-1, 4: 5e-1, 5: 5e-1}
        
        # # 21 joints in total(expect root joint)
        # self.smplx_body_parts = {
        #     0: [11, 14],                    # head
        #     1: [12, 15, 17, 19],            # left-arm
        #     2: [13, 16, 18, 20],            # right-arm
        #     3: [0, 3, 6, 9],                # left-leg
        #     4: [1, 4, 7, 10]}               # right-leg
        
        # 22 joints in total
        self.smplx_body_parts = {
            0: [12, 15],                    # head
            1: [13, 16, 18, 20],            # left-arm
            2: [14, 17, 19, 21],            # right-arm
            3: [1, 4, 7, 10],                # left-leg
            4: [2, 5, 8, 11]}               # right-leg

    def preprocess(self, x):
        # (bs, num_joints, 6) -> (bs, 6, num_joints)
        x = x.permute(0,2,1)
        return x

    def forward(self, x, global_step=None):
        if self.training and global_step is not None:
            # add noise to body part
            step = global_step // 5000
            # random select half batch
            noise_scale = float(self.epoch_noise_scale_mapping[step]) if step <= 5 else 5e-1
            batch_size = x.shape[0]
            noised_samples = np.random.randint(low=0, high=batch_size-1, size=batch_size//2)
            # random select one body part add noise
            mask_part = np.random.randint(len(self.smplx_body_parts.keys()))
            masked_joints = self.smplx_body_parts[mask_part]
            x[noised_samples][:,masked_joints] += (torch.tensor(1, dtype=torch.float32).to(x.device).uniform_() * noise_scale)

        x = self.preprocess(x)
        x = self.encoder(x)
        return x

class TokenHMRDecoder(nn.Module):
    def __init__(self,
                 input_dim=512,
                 hidden_dim=512,
                 output_dim=6,
                 upsample_step=1, 
                 downsample_step=1, 
                 num_joints=21):
        super(TokenHMRDecoder, self).__init__()
        num_tokens = (((num_joints*2//10)*10) * (2**(upsample_step)) / (2**downsample_step))
        decoder_layers = []

        decoder_layers.append(nn.Conv1d(input_dim, hidden_dim, 3, 1, 1))
        decoder_layers.append(nn.ReLU())

        print(f'Num of tokens --> {num_tokens}')
        for i in list(np.linspace(num_joints, num_tokens, 4, endpoint=False, dtype=int)[::-1]):
            decoder_layers.append(nn.Upsample(i))
            decoder_layers.append(nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1))
            decoder_layers.append(nn.ReLU())
        for i in range(downsample_step):
            out_dim = hidden_dim
            decoder_layers.append(Resnet1D(
                hidden_dim, n_depth=2, dilation_growth_rate=3, reverse_dilation=True, activation='relu', norm=False))
            decoder_layers.append(nn.Conv1d(hidden_dim, out_dim, 3, 1, 1))

        # 降维
        decoder_layers.append(nn.Conv1d(hidden_dim, output_dim, 3, 1, 1))

        self.decoder = nn.Sequential(*decoder_layers)
        self.token_num = num_tokens

    def postprocess(self, x):
        # (bs, 6, num_joints) -> (bs, num_joints, 6)
        x = x.permute(0,2,1)
        return x

    def forward(self, x):
        x = self.decoder(x)
        pred_pose = self.postprocess(x)
        return pred_pose


class TransformerPoseEncoderV1(nn.Module):
    def __init__(self,
                 input_dim=6,
                 hidden_dim=512, 
                 output_dim=512):
        super(TransformerPoseEncoderV1, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, 1, 1), 
            nn.ReLU())

        self.pose_token_embedding = nn.Embedding(80, hidden_dim)
        self.self_attn1 = TransformerBlock(hidden_dim, 8, 512)
        self.self_attn2 = TransformerBlock(hidden_dim, 8, 512)
        self.self_attn3 = TransformerBlock(hidden_dim, 8, 512)
        self.out_conv = nn.Conv1d(hidden_dim, output_dim, 3, 1, 1)

        self.pe = PositionalEncoding(512, 80 + 22)

        torch.nn.init.normal_(self.pose_token_embedding.weight, mean=0.0, std=0.02)
        self.epoch_noise_scale_mapping = {
            0: 1e-2, 1: 5e-2, 2: 1e-1, 3: 1e-1, 4: 5e-1, 5: 5e-1}
        
        # 22 joints in total
        self.smplx_body_parts = {
            0: [12, 15],                    # head
            1: [13, 16, 18, 20],            # left-arm
            2: [14, 17, 19, 21],            # right-arm
            3: [1, 4, 7, 10],                # left-leg
            4: [2, 5, 8, 11]}               # right-leg

    def preprocess(self, x):
        # (bs, num_joints, 6) -> (bs, 6, num_joints)
        x = x.permute(0,2,1)
        return x

    def forward(self, x, global_step=None):
        if self.training and global_step is not None:
            # add noise to body part
            step = global_step // 5000
            # random select half batch
            noise_scale = float(self.epoch_noise_scale_mapping[step]) if step <= 5 else 5e-1
            batch_size = x.shape[0]
            noised_samples = np.random.randint(low=0, high=batch_size-1, size=batch_size//2)
            # random select one body part add noise
            mask_part = np.random.randint(len(self.smplx_body_parts.keys()))
            masked_joints = self.smplx_body_parts[mask_part]
            x[noised_samples][:,masked_joints] += (torch.tensor(1, dtype=torch.float32).to(x.device).uniform_() * noise_scale)

        x = self.preprocess(x)
        x = self.in_conv(x) # (bs, C, num_joints)
        x = x.permute(0, 2, 1) # (bs, num_joints, C)
        pose_embedding = self.pose_token_embedding.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.concat([x, pose_embedding], dim=1)
        x = self.pe(x)
        x = self.self_attn1(x)
        x = self.self_attn2(x)
        x = self.self_attn3(x)
        x = self.out_conv(x.permute(0, 2, 1)[..., 22:])
        return x

class TransformerPoseDecoderV1(nn.Module):
    def __init__(self,
                 input_dim=512,
                 hidden_dim=512,
                 output_dim=6,
                 num_joints=21):
        super(TransformerPoseDecoderV1, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, 1, 1), 
            nn.ReLU())

        self.pose_token_embedding = nn.Embedding(num_joints, hidden_dim)
        torch.nn.init.normal_(self.pose_token_embedding.weight, mean=0.0, std=0.02)
        self.pe = PositionalEncoding(512, 80 + 22)
        self.self_attn1 = TransformerBlock(hidden_dim, 8, 512)
        self.self_attn2 = TransformerBlock(hidden_dim, 8, 512)
        self.self_attn3 = TransformerBlock(hidden_dim, 8, 512)

        self.out_conv = nn.Conv1d(hidden_dim, output_dim, 3, 1, 1)
        self.token_num = 80

    def postprocess(self, x):
        # (bs, 6, num_joints) -> (bs, num_joints, 6)
        x = x.permute(0,2,1)
        return x

    def forward(self, x):
        x = self.in_conv(x).permute(0, 2, 1)
        pose_embedding = self.pose_token_embedding.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.concat([pose_embedding, x], dim=1)
        x = self.pe(x)
        x = self.self_attn1(x)
        x = self.self_attn2(x)
        x = self.self_attn3(x).permute(0, 2, 1)
        pred_pose = self.out_conv(x[..., :22]).permute(0, 2, 1)
        return pred_pose
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=100):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        # 创建位置编码矩阵
        position = torch.arange(max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))  # (embed_dim // 2,)
        pe = torch.zeros(max_seq_len, embed_dim)  # (max_seq_len, embed_dim)

        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_len, embed_dim)

    def forward(self, x):
        """
        x: 输入张量，形状为 (batch_size, seq_len, embed_dim)
        """
        return x + self.pe[:, :x.size(1)]  # 添加位置编码

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 确保 embed_dim 可以被 num_heads 整除
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"

        # 定义线性变换函数
        self.linear_q = nn.Linear(embed_dim, embed_dim)  # 查询变换
        self.linear_k = nn.Linear(embed_dim, embed_dim)  # 键变换
        self.linear_v = nn.Linear(embed_dim, embed_dim)  # 值变换
        self.linear_out = nn.Linear(embed_dim, embed_dim)  # 输出变换

    def forward(self, x, mask=None):
        """
        x: 输入张量，形状为 (batch_size, seq_len, embed_dim)
        mask: 可选掩码，形状为 (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape

        # 线性变换得到 Q, K, V
        q = self.linear_q(x)  # (batch_size, seq_len, embed_dim)
        k = self.linear_k(x)  # (batch_size, seq_len, embed_dim)
        v = self.linear_v(x)  # (batch_size, seq_len, embed_dim)

        # 分割成多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

        # 可选：应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        # 拼接多头结果
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

        # 线性变换输出
        output = self.linear_out(attn_output)  # (batch_size, seq_len, embed_dim)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim

        # Multi-Head Self-Attention
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 残差连接 + 层归一化
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.norm1(x)  # 层归一化

        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)  # 残差连接
        x = self.norm2(x)  # 层归一化

        return x