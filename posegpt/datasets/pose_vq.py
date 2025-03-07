'''
copy from tokenhmr
'''
import torch
from torch.utils.data import Dataset
import numpy as np
import os.path as osp
from ..utils.rotation_conversions import axis_angle_to_matrix, rotvec_to_eulerangles, eulerangles_to_rotvec
from ..utils import BodyModel
from typing import List

class MixedPoseVQDataset(Dataset):
    def __init__(self, 
                 dataset_root: str, 
                 dataset_list: List[str], 
                 smpl_path: str, 
                 dataset_partition: List[float]=None, 
                 split: str='train', 
                 **kwargs) -> None:
        super().__init__()
        self.split = split
        # prepare train data
        if split == 'train':
            self.dataset_list = dataset_list
            self.dataset_partition = np.array(dataset_partition).cumsum()

            self.datasets = [PoseVQDataset(dataset, dataset_root, smpl_path, split) for dataset in dataset_list]
            self.length = max([len(dataset) for dataset in self.datasets])
        # prepare val/test data
        else:
            self.datasets = ValMixedPoseVQDataset(dataset_root, dataset_list, smpl_path, split)
            self.length = len(self.datasets)

    def __getitem__(self, index):
        if self.split == 'train':
            p = np.random.rand()
            for i in range(len(self.dataset_list)):
                if p <= self.dataset_partition[i]:
                    return self.datasets[i][index % len(self.datasets[i])]
        else:
            return self.datasets[index]

    def __len__(self):
        return self.length

class PoseVQDataset(Dataset):
    def __init__(self, 
                 dataset: str, # dataset name
                 dataset_root: str, 
                 smpl_path: str, 
                 split: str='train', ):
        super().__init__()
        assert split == 'train'
        self.split = split
        
        self.body_model = BodyModel(smpl_path)
        dataset_path = osp.join(dataset_root, split, f'{split}_{dataset}.npz')
        data_infos = np.load(dataset_path)
        self.pose_body = data_infos['pose_body']
        # self.betas = data_infos['betas']
        self.global_orient = data_infos['global_orient']

        print(f"Processing {dataset} for {split} with {self.pose_body.shape[0]} samples...")

    def __len__(self):
        return self.pose_body.shape[0]

    def __getitem__(self, index):
        item = dict()
        global_orient = self.global_orient[index] # axis angle
        if self.split == 'train':
            # add noise to global orient: front-z: -45~45, xy: -20~20
            thetax, thetay, thetaz = rotvec_to_eulerangles(torch.tensor(global_orient)[None])
            thetaz += torch.empty(1).uniform_(-torch.pi / 4, torch.pi / 4)
            thetax += torch.empty(1).uniform_(-torch.pi / 9, torch.pi / 9)
            thetay += torch.empty(1).uniform_(-torch.pi / 9, torch.pi / 9)
            global_orient = eulerangles_to_rotvec(thetax, thetay, thetaz)[0].numpy()

        pose_body = self.pose_body[index]
        pose_body = np.concatenate([global_orient, pose_body], axis=0) 
        pose_body = torch.Tensor(pose_body.reshape(-1)).float()

        body_model = self.body_model(
            root_orient=pose_body[:3][None], 
            pose_body=pose_body[3:][None])
        
        item['body_vertices'] = body_model.v[0].detach().float()
        item['body_joints'] = body_model.Jtr[0].detach().float()

        # import trimesh
        # trimesh.Trimesh(vertices=body_model.v.detach().numpy()[0], faces=body_model.f).export('p1.obj')
        # import ipdb; ipdb.set_trace()

        pose_body_rot = axis_angle_to_matrix(pose_body.view(-1,3))
        item['body_pose_rotmat'] = pose_body_rot

        return item


class ValMixedPoseVQDataset(Dataset):
    def __init__(self, 
                 dataset_root, 
                 dataset_list, 
                 smpl_path: str, 
                 split= 'val'):
        super().__init__()
        self.dataset_root = osp.join(dataset_root, split)
        self.split = split

        self.body_model = BodyModel(smpl_path)

        self.pose_body = np.empty((0,63))
        self.global_orient = np.empty((0, 3))

        for dataset in dataset_list:
            data = np.load(osp.join(self.dataset_root, f'{split}_{dataset}.npz'))
            self.pose_body = np.append(self.pose_body, data['pose_body'], axis=0)
            self.global_orient = np.append(self.global_orient, data['global_orient'], axis=0)
            # self.betas = np.append(self.betas, data['betas'], axis=0)
            print(f"Processing {dataset} for {split} with {data['pose_body'].shape[0]} samples...")

        print(f"Total number of samples {self.pose_body.shape[0]}")

    def __len__(self):
        return self.pose_body.shape[0]

    def __getitem__(self, index):
        item = dict()
        global_orient = self.global_orient[index]
        pose_body = self.pose_body[index]
        pose_body = np.concatenate([global_orient, pose_body], axis=0) 
        pose_body = torch.Tensor(pose_body.reshape(-1)).float()

        item['body_pose_axis_angle'] = pose_body.clone()

        body_model = self.body_model(
            root_orient=pose_body[:3][None], 
            pose_body=pose_body[3:][None])
        item['body_vertices'] = body_model.v[0].detach().float()
        item['body_joints'] = body_model.Jtr[0].detach().float()

        # import trimesh
        # trimesh.Trimesh(vertices=body_model.v.detach().numpy()[0], faces=body_model.f).export('p1.obj')
        # import ipdb; ipdb.set_trace()

        item['body_pose_rotmat'] = axis_angle_to_matrix(pose_body.view(-1,3))

        return item
