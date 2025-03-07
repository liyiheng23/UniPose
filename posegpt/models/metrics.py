'''
copy from posescript/motiongpt
'''
import abc
import smplx
import torch
from torchmetrics import Metric
from posegpt.utils.metric_utils import (
    calc_mpjpe, calc_pampjpe, calculate_activation_statistics_np, calculate_top_k, 
    calculate_frechet_distance_np)
from .base_module import build_model
from posegpt.utils import load_checkpoint
from typing import Dict
import pickle
from typing import Optional
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput

from ..utils import BodyModel
from ..utils.rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix
import trimesh
import numpy as np

class BaseMetric(Metric):
    def __init__(self, dist_sync_on_step=True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=False)
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("gt_text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("pred_text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gt_pose_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("pred_pose_embeddings", default=[], dist_reduce_fx=None)

        self.add_state("gt_texts", default=[], dist_reduce_fx=None)
        self.add_state("pred_texts", default=[], dist_reduce_fx=None)

        # pose reconstruction metric
        self.add_state("MPJPE", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("PAMPJPE", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("MPJRE", default=torch.tensor([0.0]), dist_reduce_fx="sum")

        # elbo metric, see posescript for detailed info.
        self.add_state("ELBO_joints", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("ELBO_vertices", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("ELBO_rot", default=torch.tensor([0.0]), dist_reduce_fx="sum")
    
    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def compute(self) -> Dict:
        pass

class PoseReconstructionMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        # h36m dataset need special treatment
        self.body_model = BodyModel('processed_dataset/smpl_models/smplh/SMPLH_NEUTRAL.npz')
        self.smpl = SMPL(
            data_dir='processed_dataset/smpl_models', 
            model_path='processed_dataset/smpl_models/smpl', 
            joint_regressor_extra='processed_dataset/smpl_models/SMPL_to_J19.pkl', 
            gender='neutral', )
        self.keypoints_list = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 43]
        self.pelvis_ind = 39

    def update(self, pred_axis_angles, keypoints_3d, **kwargs):
        gt_joints = keypoints_3d[..., :-1]
        pred_axis_angles = pred_axis_angles.view(pred_axis_angles.shape[0], -1, 3)

        # amass的global rot需要转成h36m的格式（绕x轴逆时针转90度）
        mat1 = axis_angle_to_matrix(pred_axis_angles[:, 0]).to(torch.float64)
        mat2 = torch.tensor(trimesh.transformations.rotation_matrix(np.radians(90), (1, 0, 0))[:3, :3]).to(torch.float64)
        mat = mat2 @ mat1
        pred_rotmat = axis_angle_to_matrix(pred_axis_angles) # b, j, 3, 3
        pred_rotmat[:, 0] = mat

        pred_rotmat = torch.nn.functional.pad(pred_rotmat, (0, 0, 0, 0, 0, 2), mode='constant', value=0.)
        pred_joints = self.smpl(
            global_orient=pred_rotmat[:, 0], 
            body_pose=pred_rotmat[:, 1:]).joints # b, 25, 3
        
        pred_joints = pred_joints - pred_joints[:, [self.pelvis_ind]]
        gt_joints = gt_joints - gt_joints[:, [self.pelvis_ind]]

        pred_joints = pred_joints[:, self.keypoints_list]
        gt_joints = gt_joints[:, self.keypoints_list]

        # (bs, njoint = 22, 3)
        self.count += pred_joints.shape[0]
        # avoid cuda error of DDP in pampjpe
        rst = pred_joints.detach().cpu()
        ref = gt_joints.detach().cpu()
        self.MPJPE += torch.sum(calc_mpjpe(rst, ref, align_inds=None))
        self.PAMPJPE += torch.sum(calc_pampjpe(rst, ref))

    def compute(self):
        factor = 1000.0
        metrics = dict()
        metrics["MPJPE"] = (self.MPJPE / self.count * factor)
        metrics["PAMPJPE"] = (self.PAMPJPE / self.count * factor)
        self.reset()
        return metrics

# 用SMPL的关节点当gt
class PoseReconstruction_SMPL_Metric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.body_model = BodyModel('processed_dataset/smpl_models/smplh/SMPLH_NEUTRAL.npz')

    def update(self, pred_axis_angles, gt_axis_angles, **kwargs):
        pred_axis_angles = pred_axis_angles.view(pred_axis_angles.shape[0], -1)
        gt_axis_angles = gt_axis_angles.view(gt_axis_angles.shape[0], -1)

        pred_joints = self.body_model(
            root_orient=pred_axis_angles[:, :3], 
            pose_body=pred_axis_angles[:, 3:]).Jtr # b, 25, 3
        
        gt_joints = self.body_model(
            root_orient=gt_axis_angles[:, :3], 
            pose_body=gt_axis_angles[:, 3:]).Jtr # b, 25, 3

        # (bs, njoint = 22, 3)
        self.count += pred_joints.shape[0]
        # avoid cuda error of DDP in pampjpe
        rst = pred_joints.detach().cpu()
        ref = gt_joints.detach().cpu()
        self.MPJPE += torch.sum(calc_mpjpe(rst, ref, align_inds=None))
        self.PAMPJPE += torch.sum(calc_pampjpe(rst, ref))

    def compute(self):
        factor = 1000.0
        metrics = dict()
        metrics["MPJPE"] = (self.MPJPE / self.count * factor)
        metrics["PAMPJPE"] = (self.PAMPJPE / self.count * factor)
        self.reset()
        return metrics


class Text2PoseMetric(BaseMetric):
    def __init__(self, pose_text_encoder_config=None, pose_text_encoder_ckp_path=None):
        super().__init__()
        self.top_k = 3
        self.R_size = 32

        self.pose_text_encoder = build_model(pose_text_encoder_config)
        load_checkpoint(self.pose_text_encoder, pose_text_encoder_ckp_path)
        self.pose_text_encoder.eval()
        for p in self.pose_text_encoder.parameters():
            p.reqiures_grad = False
        
        self.body_model = BodyModel('processed_dataset/smpl_models/smplh/SMPLH_NEUTRAL.npz')

    def update(self, pred_pose, gt_pose, gt_text, **kwargs):
        # pose为rotmat格式: bs, n, 3, 3
        assert pred_pose.shape[-2:] == torch.Size([3, 3])
        assert gt_pose.shape[-2:] == torch.Size([3, 3])
        self.count += pred_pose.shape[0]

        pred_pose_embd = self.pose_text_encoder.pose_encoder(pred_pose)
        gt_pose_embd = self.pose_text_encoder.pose_encoder(gt_pose)
        text_embd = self.pose_text_encoder.text_encoder(gt_text)

        self.pred_pose_embeddings.append(pred_pose_embd)
        self.gt_pose_embeddings.append(gt_pose_embd)
        self.gt_text_embeddings.append(text_embd)

        # rotmat -> axis angle
        pred_pose = matrix_to_axis_angle(pred_pose).flatten(1, 2)
        gt_pose = matrix_to_axis_angle(gt_pose).flatten(1, 2)
        pred_joints = self.body_model(root_orient=pred_pose[:, :3], pose_body=pred_pose[:, 3:]).Jtr
        gt_joints = self.body_model(root_orient=gt_pose[:, :3], pose_body=gt_pose[:, 3:]).Jtr

        # calculate elbo metrics
        pred_joints = pred_joints.detach().cpu()
        gt_joints = gt_joints.detach().cpu()
        self.MPJPE += torch.sum(calc_mpjpe(pred_joints, gt_joints, align_inds=None))
        self.PAMPJPE += torch.sum(calc_pampjpe(pred_joints, gt_joints))

    def compute(self):
        count = self.count
        factor = 1000
        metrics = dict()
        # Cat cached batches and shuffle
        shuffle_idx = torch.randperm(count)

        all_gt_text_embds = torch.cat(self.gt_text_embeddings, dim=0).cpu()[shuffle_idx, :]
        all_gt_pose_embds = torch.cat(self.gt_pose_embeddings, dim=0).cpu()[shuffle_idx, :]
        all_pred_pose_embds = torch.cat(self.pred_pose_embeddings, dim=0).cpu()[shuffle_idx, :]

        # R_precision
        assert count > self.R_size
        self.matching_score, top_k_mat = calculate_R_precision(
            all_gt_text_embds, all_pred_pose_embds, self.R_size, self.top_k)
        R_count = torch.div(count, self.R_size, rounding_mode='floor') * self.R_size
        metrics["Matching_Score"] = self.matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_Precision_Top{str(k+1)}"] = top_k_mat[k] / R_count

        # FID
        mu, cov = calculate_activation_statistics_np(all_pred_pose_embds.numpy())
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gt_pose_embds.numpy())
        metrics['FID'] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # pose error
        metrics["MPJPE"] = self.MPJPE / self.count * factor
        # metrics['MPJRE'] = self.MPJRE / self.count * factor
        metrics["PAMPJPE"] = self.PAMPJPE / self.count * factor

        # compute text2pose, pose2text recall
        k_values = [1, 5, 10, 20]
        t2p_scores = all_gt_text_embds.mm(all_pred_pose_embds.t()).cpu() 

        t2p_top_k_mat = calculate_A2B_topk_recalls(
            all_gt_text_embds, all_pred_pose_embds, k_values, t2p_scores)
        p2t_top_k_mat = calculate_A2B_topk_recalls(
            all_pred_pose_embds, all_gt_text_embds, k_values, t2p_scores.t())
        
        mRecall = 0
        for k in k_values:
            metrics[f'Text2Pose_Top{k}'] = t2p_top_k_mat[k - 1]
            metrics[f'Pose2Text_Top{k}'] = p2t_top_k_mat[k - 1]
            mRecall += (t2p_top_k_mat[k - 1] + p2t_top_k_mat[k - 1])
            
        mRecall /= (2 * len(k_values))
        metrics['mRecall'] = mRecall

        # metrics = {f'METRIC/{k}': v for k, v in metrics.items()}
        metrics = {f'{k}': v for k, v in metrics.items()}

        self.reset()
        return metrics

class Pose2TextMetric(BaseMetric):
    def __init__(self, pose_text_encoder_config=None, pose_text_encoder_ckp_path=None):
        super().__init__()
        # for calculating R-precision
        self.top_k = 3 
        self.R_size = 32
        # for calculating retrieval metric
        self.k_values = [1, 5, 10, 20] 
        self.nlp_evaluator = NLPEvaluator(metric_list=['bleu', 'rouge', 'meteor'])
        self.pose_text_encoder = build_model(pose_text_encoder_config)
        load_checkpoint(self.pose_text_encoder, pose_text_encoder_ckp_path)
        self.pose_text_encoder.eval()
        for p in self.pose_text_encoder.parameters():
            p.reqiures_grad = False

    def update(self, gt_pose, pred_text, gt_text, **kwargs):
        # pose为rotmat格式: bs, n, 3, 3
        assert gt_pose.shape[-2:] == torch.Size([3, 3])
        self.count += len(pred_text)
        self.pred_texts.extend(pred_text)
        self.gt_texts.extend(gt_text)
        pose_embd = self.pose_text_encoder.pose_encoder(gt_pose)
        pred_text_embd = self.pose_text_encoder.text_encoder(pred_text)

        self.gt_pose_embeddings.append(pose_embd)
        self.pred_text_embeddings.append(pred_text_embd)

    def compute(self):
        count = self.count
        metrics = dict()
        # Cat cached batches and shuffle
        # torch.manual_seed(3407)
        shuffle_idx = torch.randperm(count)

        all_gt_pose_embds = torch.cat(self.gt_pose_embeddings, dim=0).cpu()[shuffle_idx, :]
        all_pred_text_embds = torch.cat(self.pred_text_embeddings, dim=0).cpu()[shuffle_idx, :]
        # R_precision
        assert count > self.R_size
        matching_score, top_k_mat = calculate_R_precision(
            all_pred_text_embds, all_gt_pose_embds, self.R_size, self.top_k)
        R_count = torch.div(count, self.R_size, rounding_mode='floor') * self.R_size
        metrics["Matching_Score"] = matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_Precision_Top{str(k+1)}"] = top_k_mat[k] / R_count

        # compute text2pose, pose2text recall
        k_values = [1, 5, 10, 20]

        t2p_scores = all_pred_text_embds.mm(all_gt_pose_embds.t()).cpu() 

        t2p_top_k_mat = calculate_A2B_topk_recalls(
            all_pred_text_embds, all_gt_pose_embds, k_values, t2p_scores)
        p2t_top_k_mat = calculate_A2B_topk_recalls(
            all_gt_pose_embds, all_pred_text_embds, k_values, t2p_scores.t())
        
        mRecall = 0
        for k in k_values:
            metrics[f'Text2Pose_Top{k}'] = t2p_top_k_mat[k - 1]
            metrics[f'Pose2Text_Top{k}'] = p2t_top_k_mat[k - 1]
            mRecall += (t2p_top_k_mat[k - 1] + p2t_top_k_mat[k - 1])
            
        mRecall /= (2 * len(k_values))
        metrics['mRecall'] = mRecall
        nlp_metric = self.nlp_evaluator(predictions=self.pred_texts, references=self.gt_texts)

        metrics['BLEU-4'] = nlp_metric['bleu']['bleu']
        metrics['ROUGE-L'] = nlp_metric['rouge']['rougeL']
        metrics['METEOR'] = nlp_metric['meteor']['meteor']

        # metrics = {f'METRIC/{k}': v for k, v in metrics.items()}
        metrics = {f'{k}': v for k, v in metrics.items()}
        self.reset()
        return metrics

class RetrievalMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.R_size = 32
        self.top_k = [1, 5, 10]

    def update(self, pred_pose_embd, pred_text_embd):
        self.count += pred_pose_embd.shape[0]
        self.gt_pose_embeddings.append(pred_pose_embd)
        self.gt_text_embeddings.append(pred_text_embd)

    def compute(self):
        metrics = dict()
        # Cat cached batches and shuffle
        shuffle_idx = torch.randperm(self.count)

        all_pose_embds = torch.cat(self.gt_pose_embeddings, dim=0).cpu()[shuffle_idx, :]
        all_text_embds = torch.cat(self.gt_text_embeddings, dim=0).cpu()[shuffle_idx, :]

        # R_precision
        assert self.count > self.R_size
        matching_score, top_k_mat = calculate_R_precision(
            all_text_embds, all_pose_embds, self.R_size, top_k=3)
        R_count = torch.div(self.count, self.R_size, rounding_mode='floor') * self.R_size
        metrics["Matching_Score"] = matching_score / R_count
        for k in range(3):
            metrics[f"R_Precision_Top{str(k+1)}"] = top_k_mat[k] / R_count
        
        # compute text2pose, pose2text recall
        t2p_scores = all_text_embds.mm(all_pose_embds.t()).cpu() 

        t2p_top_k_mat = calculate_A2B_topk_recalls(
            all_text_embds, all_pose_embds, self.top_k, t2p_scores)
        p2t_top_k_mat = calculate_A2B_topk_recalls(
            all_pose_embds, all_text_embds, self.top_k, t2p_scores.t())
        
        mRecall = 0
        for k in self.top_k:
            metrics[f'Text2Pose_Top{k}'] = t2p_top_k_mat[k - 1]
            metrics[f'Pose2Text_Top{k}'] = p2t_top_k_mat[k - 1]
            mRecall += (t2p_top_k_mat[k - 1] + p2t_top_k_mat[k - 1])
            
        mRecall /= (2 * len(self.top_k))
        metrics['mRecall'] = mRecall
        
        # metrics = {f'METRIC/{k}': v for k, v in metrics.items()}
        metrics = {f'{k}': v for k, v in metrics.items()}
        self.reset()
        return metrics
    
class PoseEditMetric(BaseMetric):
    def __init__(self, pose_text_encoder_config=None, pose_text_encoder_ckp_path=None):
        super().__init__()
        self.top_k = 3
        self.R_size = 32

        self.pose_text_encoder = build_model(pose_text_encoder_config)
        load_checkpoint(self.pose_text_encoder, pose_text_encoder_ckp_path)
        self.pose_text_encoder.eval()
        for p in self.pose_text_encoder.parameters():
            p.reqiures_grad = False
        
        self.body_model = BodyModel('processed_dataset/smpl_models/smplh/SMPLH_NEUTRAL.npz')

    def update(self, pred_pose, gt_pose, **kwargs):
        # pose为rotmat格式: bs, n, 3, 3
        assert pred_pose.shape[-2:] == torch.Size([3, 3])
        assert gt_pose.shape[-2:] == torch.Size([3, 3])
        self.count += pred_pose.shape[0]

        pred_pose_embd = self.pose_text_encoder.pose_encoder(pred_pose)
        gt_pose_embd = self.pose_text_encoder.pose_encoder(gt_pose)
        self.pred_pose_embeddings.append(pred_pose_embd)
        self.gt_pose_embeddings.append(gt_pose_embd)

        # rotmat -> axis angle
        pred_pose = matrix_to_axis_angle(pred_pose).flatten(1, 2)
        gt_pose = matrix_to_axis_angle(gt_pose).flatten(1, 2)
        pred_joints = self.body_model(root_orient=pred_pose[:, :3], pose_body=pred_pose[:, 3:]).Jtr.detach().cpu()
        gt_joints = self.body_model(root_orient=gt_pose[:, :3], pose_body=gt_pose[:, 3:]).Jtr.detach().cpu()
        self.MPJPE += torch.sum(calc_mpjpe(pred_joints, gt_joints, align_inds=None))
        self.PAMPJPE += torch.sum(calc_pampjpe(pred_joints, gt_joints))

    def compute(self):
        factor = 1000
        metrics = dict()

        all_gt_pose_embds = torch.cat(self.gt_pose_embeddings, dim=0).cpu()
        all_pred_pose_embds = torch.cat(self.pred_pose_embeddings, dim=0).cpu()

        # FID
        mu, cov = calculate_activation_statistics_np(all_pred_pose_embds.numpy())
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gt_pose_embds.numpy())
        metrics['FID'] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # pose error
        metrics["MPJPE"] = self.MPJPE / self.count * factor
        metrics["PAMPJPE"] = self.PAMPJPE / self.count * factor

        metrics = {f'{k}': v for k, v in metrics.items()}

        self.reset()
        return metrics

class PoseDifferenceMetric(Pose2TextMetric):
    def update(self, gt_pose, pred_text, gt_text, **kwargs):
        # pose为rotmat格式: 2(poseA & poseB), bs, n, 3, 3
        assert gt_pose.shape[-2:] == torch.Size([3, 3])
        self.count += len(pred_text)
        self.pred_texts.extend(pred_text)
        self.gt_texts.extend(gt_text)
        pose_embd = self.pose_text_encoder.pose_encoder(gt_pose[0], gt_pose[1])
        pred_text_embd = self.pose_text_encoder.text_encoder(pred_text)

        self.gt_pose_embeddings.append(pose_embd)
        self.pred_text_embeddings.append(pred_text_embd)

# Compute r-precision NOTE: 这个是参考的motionGPT，与posescript的实现有出入
def calculate_R_precision(embdA, embdB, R_size, top_k):
    top_k_mat = torch.zeros((top_k, ))
    count = embdA.shape[0]
    matching_score = 0
    # 计算cos sim
    for i in range(count // R_size):
        groupA = embdA[i * R_size:(i + 1) * R_size] # bs, c
        groupB = embdB[i * R_size:(i + 1) * R_size] # bs, c
        scores = groupA.mm(groupB.t()) # bs, bs
        matching_score += scores.trace()
        argsmax = torch.argsort(scores, dim=1, descending=True)
        top_k_mat += calculate_top_k(argsmax, top_k).sum(axis=0)
    return matching_score, top_k_mat

def calculate_A2B_topk_recalls(embdA, embdB, k_values, A2B_scores=None):
    if A2B_scores is None:
        A2B_scores = embdA.mm(embdB.t()).cpu()
    count = A2B_scores.shape[0]
    # 计算cos sim
    A2B_argsmax = torch.argsort(A2B_scores, dim=1, descending=True)
    A2B_top_k_mat = calculate_top_k(A2B_argsmax, top_k=k_values[-1]).sum(axis=0)
    A2B_top_k_mat = A2B_top_k_mat / count
    return A2B_top_k_mat

# evaluate库里面的load metric速度太慢
class NLPEvaluator:
    def __init__(self, metric_list) -> None:
        self.metric_list = metric_list
        for metric in metric_list:
            if metric == 'bleu':
                from extra_libs.evaluate.metrics.bleu.bleu import Bleu
                self.bleu = Bleu()
            elif metric == 'rouge':
                from extra_libs.evaluate.metrics.rouge.rouge import Rouge
                self.rouge = Rouge()
            elif metric == 'bertscore':
                from extra_libs.evaluate.metrics.bertscore.bertscore import BERTScore
                self.bertscore = BERTScore()
            elif metric == 'meteor':
                from extra_libs.evaluate.metrics.meteor.meteor import Meteor
                self.meteor = Meteor()
            else:
                raise NotImplementedError

    def compute(self, predictions=None, references=None):
        metrics = dict()
        for metric in self.metric_list:
            metrics[metric] = getattr(self, metric).compute(
                predictions=predictions, references=references)
        return metrics
    
    def __call__(self, predictions=None, references=None):
        return self.compute(predictions=predictions, references=references)


class SMPL(smplx.SMPLLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, update_hips: bool = False, **kwargs):
        """
        Extension of the official SMPL implementation to support more joints.
        Args:
            Same as SMPLLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super().__init__(*args, **kwargs)
        smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                            7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
            
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))
        self.update_hips = update_hips

    def forward(self, *args, **kwargs) -> SMPLOutput:
        """
        Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
        """
        smpl_output = super().forward(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        if self.update_hips:
            joints[:,[9,12]] = joints[:,[9,12]] + \
                0.25*(joints[:,[9,12]]-joints[:,[12,9]]) + \
                0.5*(joints[:,[8]] - 0.5*(joints[:,[9,12]] + joints[:,[12,9]]))
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        smpl_output.joints = joints
        return smpl_output
