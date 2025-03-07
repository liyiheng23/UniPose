import pickle as pkl
import random
import torch
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset
import os.path as osp
import json
from posegpt.utils.geometry_conver import axis_angle_to_matrix

class RetrievalPoseScriptDataset(Dataset):
    def __init__(
            self, 
            split='train', 
            dataset_root: str=None, 
            ann_file: str=None, 
            # during inference stage, appoint a fixed caption id
            caption_selection_id=0, 
            stage='pretrain', 
            **kwargs) -> None:
        super().__init__()
        assert stage in ['pretrain', 'finetune']
        self.stage = stage
        self.split = split
        self.caption_selection_id = caption_selection_id
        print('Load posescript data...')
        with open(ann_file, 'rb') as f:
            data_infos = pkl.load(f)

        if split == 'train-val':
            with open(osp.join(dataset_root, f'train_ids_100k.json')) as f:
                data_ids = json.load(f)
            with open(osp.join(dataset_root, f'val_ids_100k.json')) as f:
                data_ids.extend(json.load(f))
        else:
            with open(osp.join(dataset_root, f'{split}_ids_100k.json')) as f:
                data_ids = json.load(f)
                
        # finetune on human annotated data
        if self.stage == 'finetune':
            with open(osp.join(dataset_root, 'posescript_human_6293.json')) as f:
                human_ids = [int(k) for k, _ in json.load(f).items()]
                self.data_infos = [data_infos[i] for i in human_ids if i in data_ids]
        else:
            self.data_infos = [data_infos[i] for i in data_ids]

    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, index):
        # 取数据
        data_info = self.data_infos[index]
        caption_list = data_info['captions']

        # 处理pose, 获取body pose的vec表达和6d matrix表达
        global_orient = torch.from_numpy(data_info['global_orient'])
        body_pose = torch.from_numpy(data_info['body_pose'])
        pose = torch.cat([global_orient, body_pose], dim=0)
        body_pose_rotmat = axis_angle_to_matrix(pose[:66].view(-1, 3))

        if self.split == 'train' and self.stage == 'pretrain':
            caption_selection_id = random.choice(range(3))
        else:
            caption_selection_id = self.caption_selection_id

        return dict(
            body_pose_rotmat=body_pose_rotmat, #  J, 3, 3
            caption=caption_list[caption_selection_id])

class RetrievalPoseFixDataset(Dataset):
    def __init__(self, 
                 split='train', 
                 dataset_root=None, 
                 ann_file=None, 
                 # during inference stage, appoint a fixed caption id
                 caption_selection_id=0, 
                 stage='pretrain', 
                 **kwargs) -> None:
        super().__init__()
        assert stage in ['pretrain', 'finetune']
        self.stage = stage
        self.split = split
        self.caption_selection_id = caption_selection_id
        print('Load posefix data...')
        with open(ann_file, 'rb') as f:
            data_infos = pkl.load(f)
        
        if split == 'train-val':
            with open(osp.join(dataset_root, f'train_in_sequence_pair_ids.json')) as f:
                data_ids = json.load(f)
            with open(osp.join(dataset_root, f'train_out_sequence_pair_ids.json')) as f:
                data_ids.extend(json.load(f))
            with open(osp.join(dataset_root, f'val_in_sequence_pair_ids.json')) as f:
                data_ids.extend(json.load(f))
            with open(osp.join(dataset_root, f'val_out_sequence_pair_ids.json')) as f:
                data_ids.extend(json.load(f))
        else:
            with open(osp.join(dataset_root, f'{split}_in_sequence_pair_ids.json')) as f:
                data_ids = json.load(f)
            with open(osp.join(dataset_root, f'{split}_out_sequence_pair_ids.json')) as f:
                data_ids.extend(json.load(f))
        # train or test on human annotated data
        if caption_selection_id == 3:
            with open(osp.join(dataset_root, 'posefix_human_6157.json')) as f:
                human_ids = [int(k) for k, _ in json.load(f).items()]
                self.data_infos = [data_infos[i] for i in human_ids if i in data_ids]
        else:
            self.data_infos = [data_infos[i] for i in data_ids]

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        data_info = self.data_infos[index]

        global_orientA = torch.from_numpy(data_info['global_orientA'])
        global_orientB = torch.from_numpy(data_info['global_orientB'])

        body_poseA = torch.from_numpy(data_info['body_poseA'])
        body_poseB = torch.from_numpy(data_info['body_poseB'])

        pose_A = torch.cat([global_orientA, body_poseA], dim=0)
        pose_B = torch.cat([global_orientB, body_poseB], dim=0)

        caption_list = data_info['captions']
        body_poseA_rotmat = axis_angle_to_matrix(pose_A[:66].view(-1, 3))
        body_poseB_rotmat = axis_angle_to_matrix(pose_B[:66].view(-1, 3))

        if self.split == 'train' and self.stage == 'pretrain':
            caption_selection_id = random.choice(range(3))
        else:
            caption_selection_id = self.caption_selection_id

        return dict(
            body_poseA_rotmat=body_poseA_rotmat, # J, 3, 3
            body_poseB_rotmat=body_poseB_rotmat,
            caption=caption_list[caption_selection_id])
