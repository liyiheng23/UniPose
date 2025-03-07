import sys
sys.path.append('/home/liyiheng/codes/posegpt')
import pickle as pkl
from posegpt.utils import BodyModel
import numpy as np
import torch
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

smpl_model_path = '/home/human/codes/liyiheng/codes/posegpt/processed_dataset/smpl_models/smplx/SMPLX_NEUTRAL.npz'

def vis_mesh(pose=None, pose_body=None, global_orient=None, save_path='p.obj'):
    if pose is not None:
        pose = torch.tensor(pose).to(torch.float64)
        pose_body = pose[None, 3:66]
        root_orient = pose[None, :3]
    else:
        pose_body = pose_body
        root_orient = global_orient
    smpl = BodyModel(smpl_model_path, dtype=torch.float64)
    p1 = smpl.forward(pose_body=pose_body, root_orient=root_orient)
    trimesh.Trimesh(vertices=p1.v.detach().numpy()[0], faces=smpl.f).export(save_path)
