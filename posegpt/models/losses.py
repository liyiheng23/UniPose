import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from posegpt.utils import BodyModel

class GPTLoss(nn.Module):
    def __init__(self, lambda_cls):
        super().__init__()
        self.lambda_cls = lambda_cls

        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer('loss', torch.tensor(0.0))

        self.loss_func = nn.Identity()
    
    def update(self, outputs):
        self.count += 1
        loss = outputs.loss

        self.loss += loss.detach()
        return loss

    def compute(self):
        '''Compute the losses and return a dictionary with the losses.'''
        count = self.count
        # Loss dictionary
        # loss_dict = {'LOSS/loss': self.loss / count}
        loss_dict = {'loss': self.loss / count}
        # Reset the losses
        self.reset()
        return loss_dict

    def reset(self):
        device = getattr(self, 'count').device
        setattr(self, 'count', torch.tensor(0.0, device=device))
        setattr(self, 'loss', torch.tensor(0.0, device=device))


class PoseVQLoss(nn.Module):
    def __init__(self, lambda_pose, lambda_joints, lambda_vertices, lambda_commit) -> None:
        super().__init__()

        self.lambda_pose = lambda_pose
        self.lambda_joints = lambda_joints
        self.lambda_vertices = lambda_vertices
        self.lambda_commit = lambda_commit

        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer('recons_pose_loss', torch.tensor(0.0))
        self.register_buffer('recons_joints_loss', torch.tensor(0.0))
        self.register_buffer('recons_vertices_loss', torch.tensor(0.0))
        self.register_buffer('vq_commit_loss', torch.tensor(0.0))
        
        self.register_buffer('total_loss', torch.tensor(0.0))

        self.losses = ['recons_pose_loss', 'recons_joints_loss', 'vq_commit_loss', 'recons_vertices_loss', 'total_loss']

        self.recons_pose_loss_func = nn.MSELoss()
        self.recons_joints_loss_func = nn.MSELoss()

        vertex_weights = self.calculate_vertex_weights()
        self.recons_vertices_loss_func = WeightedMSE(vertex_weights)

        self.vq_commit_loss_func = nn.Identity()

    def calculate_vertex_weights(self):
        body_model = BodyModel('processed_dataset/smpl_models/smplh/SMPLH_NEUTRAL.npz')
        vertices, faces = body_model.init_v_template[0].detach().numpy(), body_model.f.detach().numpy()
        v1, v2, v3 = vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]
        cross_product = np.cross(v2 - v1, v3 - v1)
        tri_area = 0.5 * np.linalg.norm(cross_product, axis=1)
        norm_triangle_area = (tri_area - np.min(tri_area)) / (np.max(tri_area) - np.min(tri_area))
        vertex_weights = np.zeros((vertices.shape[0], 1))
        for i, face in enumerate(faces):
            for vertex_index in face:
                vertex_weights[vertex_index] += norm_triangle_area[i]
        return np.repeat(vertex_weights, 3, axis=1)

    def update(self, pred_dicts, gt_dicts):
        total_loss = 0

        # reconstruction loss
        recons_pose_loss = self.recons_pose_loss_func(
            pred_dicts['pred_body_pose_rotmat'], gt_dicts['body_pose_rotmat']) * self.lambda_pose
        recons_vertices_loss = self.recons_vertices_loss_func(
            pred_dicts['pred_body_vertices'], gt_dicts['body_vertices']) * self.lambda_vertices
        recons_joints_loss = self.recons_joints_loss_func(
            pred_dicts['pred_body_joints'][:, :22], gt_dicts['body_joints'][:, :22]) * self.lambda_joints
        vq_commit_loss = self.vq_commit_loss_func(pred_dicts['loss_commit']) * self.lambda_commit
        
        total_loss = recons_pose_loss + recons_joints_loss + recons_vertices_loss + vq_commit_loss

        self.total_loss += total_loss.detach()
        self.recons_pose_loss += recons_pose_loss.detach()
        self.recons_joints_loss += recons_joints_loss.detach()
        self.recons_vertices_loss += recons_vertices_loss.detach()
        self.vq_commit_loss += vq_commit_loss.detach()

        self.count += 1
        return total_loss

    def compute(self):
        '''Compute the losses and return a dictionary with the losses.'''
        count = self.count
        # Loss dictionary
        # loss_dict = {f'LOSS/{loss_name}': getattr(self, loss_name).item()/count for loss_name in self.losses}
        loss_dict = {f'{loss_name}': getattr(self, loss_name).item()/count for loss_name in self.losses}
        # Reset the losses
        self.reset()
        return loss_dict

    def reset(self):
        device = getattr(self, 'count').device
        setattr(self, 'count', torch.tensor(0.0, device=device))

        for loss in self.losses:
            setattr(self, loss, torch.tensor(0.0, device=device))


class TextPoseMatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer('loss', torch.tensor(0.0))
        # Loss temperature
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10,)))
        self.loss_weight.requires_grad = True
    
    def update(self, pose_embd, text_embd):
        self.count += 1
        scores_t2p = text_embd.mm(pose_embd.t()) * self.loss_weight
        loss_t2p = self.CE_loss(scores_t2p)
        loss_p2t = self.CE_loss(scores_t2p.t())
        loss = (loss_p2t + loss_t2p) / 2.0
        self.loss += loss.detach()
        return loss

    def compute(self):
        '''Compute the losses and return a dictionary with the losses.'''
        count = self.count
        # Loss dictionary
        # loss_dict = {'LOSS/loss': self.loss / count}
        loss_dict = {'loss': self.loss / count}
        # Reset the losses
        self.reset()
        return loss_dict

    def reset(self):
        '''Reset to 0.'''
        setattr(self, 'loss', torch.tensor(0.0, device=getattr(self, 'loss').device))
        setattr(self, "count", torch.tensor(0.0, device=getattr(self, "count").device))
    
    def CE_loss(self, scores):
        '''
        cross entropy loss
        copy from posescript BBC Loss
        '''
        GT_labels = torch.arange(scores.shape[0], device=scores.device).long()
        loss = F.cross_entropy(scores, GT_labels) # mean reduction
        return loss

class WeightedMSE(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super(WeightedMSE, self).__init__()

        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32).cuda())
        self.reduction = reduction

    def forward(self, pred, target):
        if self.reduction == 'mean':
            return torch.mean(self.weights * (pred - target) ** 2)
