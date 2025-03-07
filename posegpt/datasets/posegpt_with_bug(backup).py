import torch
import cv2
import json
import random

from torch.utils.data import Dataset
from typing import Dict
import pickle as pkl
from posegpt.utils.geometry_conver import axis_angle_to_matrix
import numpy as np
import os.path as osp
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from torchvision.transforms.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class PoseGPTDataset(Dataset):
    def __init__(
            self, 
            image2pose_dataset: Dict=None, 
            image2text_dataset: Dict=None, 
            image_difference_dataset: Dict=None, 
            text2pose_dataset: Dict=None, 
            pose2text_dataset: Dict=None, 
            pose_edit_dataset: Dict=None, 
            pose_difference_dataset: Dict=None, 
            image_processor=None, 
            **kwargs) -> None:
        super().__init__()

        self.datasets = [
            Image2PoseDataset(**image2pose_dataset), 
            Image2TextDataset(**image2text_dataset), 
            ImageDifferenceDataset(**image_difference_dataset), 
            PoseScriptDataset(**text2pose_dataset), 
            PoseScriptDataset(**pose2text_dataset), 
            PoseFixDataset(**pose_edit_dataset), 
            PoseFixDataset(**pose_difference_dataset)
        ]
        self.data_sizes = np.array([len(dataset) for dataset in self.datasets]).cumsum()

        # set image processor
        for dataset in self.datasets:
            if isinstance(dataset, (Image2PoseDataset, Image2TextDataset, ImageDifferenceDataset)):
                setattr(dataset, 'image_processor', image_processor)
            if isinstance(dataset, Image2PoseDataset):
                # add hmr processor
                setattr(dataset, 'hmr_processor', hmr_transform(n_px=256))

    def __len__(self):
        return self.data_sizes[-1]

    def __getitem__(self, index):
        for i, dataset in enumerate(self.datasets):
            if index < self.data_sizes[i]:
                if i > 0:
                    index = index - self.data_sizes[i - 1]
                return dataset[index]
           
class BaseDataset(Dataset):
    def __init__(
            self, 
            split='train', 
            stage='pretrain', 
            ann_file: str=None, 
            dataset_root: str=None, 
            training=True, # training stage
            # for pose2text, pose_diff, image2text, image_diff task
            # during inference stage, appoint a fixed caption id
            caption_selection_id: int=0, 
            instruction_finetune=False, 
            task_name=None):
        self.split = split
        self.training = training 
        assert stage in ['pretrain', 'finetune']
        self.stage = stage
        if task_name is not None:
            if instruction_finetune:
                from scripts.instructions.finetune_instructions import instructions
            else:
                from scripts.instructions.pretrain_instructions import instructions
            self.task = instructions[task_name]

        self.caption_selection_id = caption_selection_id
        self.dataset_root = dataset_root
        if ann_file is not None:
            with open(ann_file, 'rb') as f:
                self.data_infos = pkl.load(f)

class Image2PoseDataset(BaseDataset):
    def __init__(self, 
                 dataset_list: List[Tuple[str, float]]=None, 
                 data_length=None, 
                 task_name=None, **kwargs) -> None:
        super().__init__(task_name=task_name, **kwargs)
        if task_name is None: 
            self.length = 0
            return
        self.length = data_length

        # sample data by ratio
        weights = np.array([item[1] for item in dataset_list])
        self.weights = (weights / weights.sum()).cumsum()

        self.dataset_list = []
        data_sizes = []
        print('Load image data...')
        for dataset_name, _ in dataset_list:
            data = dict(np.load(osp.join(self.dataset_root, f'{dataset_name}.npz')))
            self.dataset_list.append(data)
            data_sizes.append(data['img_path'].shape[0])
        self.data_sizes = np.array(data_sizes).cumsum()

        if self.split != 'train': 
            # only inference 1 dataset once.
            assert len(self.dataset_list) == 1
            self.length = self.dataset_list[0]['img_path'].shape[0]
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.dataset_list)):
            if p <= self.weights[i]:
                dataset = self.dataset_list[i]
                break
        dataset_size = dataset['img_path'].shape[0]
        index = index % dataset_size
        item = self.getitem(dataset, index)
        image = Image.open(osp.join(self.dataset_root, dataset['img_path'][index]))
        image = self.hmr_processor(image)
        item['hmr_imageA'] = image
        item['hmr_imageB'] = None

        return item

    def getitem(self, dataset, index):
        assert hasattr(self, 'image_processor')
        task = deepcopy(self.task)
        img_path = dataset['img_path'][index]

        image = cv2.imread(osp.join(self.dataset_root, img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        return_dict = dict()
        if self.training:
            # 训练数据必须有global_orient属性
            global_orient = torch.from_numpy(dataset['global_orient'][index])
            body_pose = torch.from_numpy(dataset['body_pose'][index])
            pose = torch.cat([global_orient, body_pose], dim=0)
            body_poseA_rotmat = axis_angle_to_matrix(pose[:66].view(-1, 3))
        else:
            # ! only used in h36m val set
            keypoints_3d = torch.from_numpy(dataset['keypoints_3d'][index])
            if keypoints_3d.size(0) == 19:
                keypoints_3d = torch.nn.functional.pad(keypoints_3d, (0, 0, 44-19, 0), mode='constant', value=0)
            body_poseA_rotmat = torch.zeros(22, 3, 3)
            return_dict['keypoints_3d'] = keypoints_3d

        task['input'] = random.choice(task['input'])
        task['output'] = random.choice(task['output'])

        return_dict.update(
            imageA=image, 
            imageB=None, # torch.zeros_like(image), 
            imgA_path=osp.join(self.dataset_root, img_path), 
            imgB_path='',

            task=task, 
            caption='', 
            body_poseA_rotmat=body_poseA_rotmat, 
            body_poseB_rotmat=torch.zeros_like(body_poseA_rotmat), 
        )
        return return_dict

class Image2TextDataset(BaseDataset):
    def __init__(self, 
                 ann_file: str=None, 
                 task_name=None, 
                 **kwargs) -> None:
        super().__init__(ann_file=ann_file, task_name=task_name, **kwargs)
        if task_name is None:
            self.data_infos = []
            return
        
        # finetune on human annotated data
        if self.stage == 'finetune':
            self.data_infos = [self.data_infos[i] for i in range(len(self.data_infos)) 
                               if len(self.data_infos[i]['captions']) == 4]

    def __len__(self):
        return len(self.data_infos) 

    def __getitem__(self, index):
        assert hasattr(self, 'image_processor')
        data_info = self.data_infos[index]
        task = deepcopy(self.task)
        item = dict()

        global_orient = torch.from_numpy(data_info[f'global_orient'])
        body_pose = torch.from_numpy(data_info[f'body_pose'])
        pose = torch.cat([global_orient, body_pose], dim=0)
        img_path = osp.join(self.dataset_root, data_info[f'img_path'])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        body_pose_rotmat = axis_angle_to_matrix(pose[:66].view(-1, 3))
        item[f'body_poseA_rotmat'] = body_pose_rotmat # J, 3, 3
        item[f'body_poseB_rotmat'] = torch.zeros_like(body_pose_rotmat)
        item[f'imageA'] = image
        item[f'imageB'] = None
        item[f'imgA_path'] = img_path
        item[f'imgB_path'] = ''

        caption_list = data_info['captions']
        if self.split == 'train' and self.stage == 'pretrain':
            select_idx = random.choice(range(3))
        else:
            select_idx = self.caption_selection_id
        task['input'] = random.choice(task['input'])
        task['output'] = random.choice(task['output'])
        item['caption'] = caption_list[select_idx]
        item['task'] = task
        return item

class ImageDifferenceDataset(BaseDataset):
    '''
    支持pose-edit任务
    '''
    def __init__(self, 
                 ann_file: str=None, 
                 task_name=None, 
                 **kwargs) -> None:
        super().__init__(ann_file=ann_file, task_name=task_name, **kwargs)
        if task_name is None:
            self.data_infos = []
            return
        # finetune on human annotated data
        if self.stage == 'finetune':
            self.data_infos = [self.data_infos[i] for i in range(len(self.data_infos))
                                if len(self.data_infos[i]['captions']) == 4]


    def __len__(self):
        return len(self.data_infos) 

    def __getitem__(self, index):
        assert hasattr(self, 'image_processor')
        data_info = self.data_infos[index]
        task = deepcopy(self.task)
        item = dict()

        for pair in ['A', 'B']:
            global_orient = torch.from_numpy(data_info[f'global_orient{pair}'])
            body_pose = torch.from_numpy(data_info[f'body_pose{pair}'])
            pose = torch.cat([global_orient, body_pose], dim=0)
            img_path = osp.join(self.dataset_root, data_info[f'img{pair}_path'])
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            body_pose_rotmat = axis_angle_to_matrix(pose[:66].view(-1, 3))
            item[f'body_pose{pair}_rotmat'] = body_pose_rotmat # J, 3, 3
            item[f'image{pair}'] = image
            item[f'img{pair}_path'] = img_path

        caption_list = data_info['captions']
        if self.split == 'train' and self.stage == 'pretrain':
            select_idx = random.choice(range(3))
        else:
            select_idx = self.caption_selection_id
        task['input'] = random.choice(task['input'])
        task['output'] = random.choice(task['output'])
        item['caption'] = caption_list[select_idx]
        item['task'] = task
        return item

class PoseScriptDataset(BaseDataset):
    '''
    支持pose2text, text2pose任务
    '''
    def __init__(self, 
                 ann_file: str=None, 
                 task_name=None, 
                 **kwargs) -> None:
        super().__init__(ann_file=ann_file, task_name=task_name, **kwargs)
        if task_name is None:
            self.data_infos = []
            return
        
        if self.split == 'train-val':
            data_ids = []
            for split in ['train', 'val']:
                with open(osp.join(self.dataset_root, f'{split}_ids_100k.json')) as f:
                    data_ids.extend(json.load(f))
        else:
            with open(osp.join(self.dataset_root, f'{self.split}_ids_100k.json')) as f:
                data_ids = json.load(f)
        
        # finetune on human annotated data
        if self.stage == 'finetune':
            with open(osp.join(self.dataset_root, 'posescript_human_6293.json')) as f:
                human_ids = [int(k) for k, _ in json.load(f).items()]
                self.data_infos = [self.data_infos[i] for i in human_ids if i in data_ids]
        else:
            self.data_infos = [self.data_infos[i] for i in data_ids]

    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, index):
        # 取数据
        data_info = self.data_infos[index]
        task = deepcopy(self.task)

        caption_list = data_info['captions']
        if self.split == 'train' and self.stage == 'pretrain':
            select_idx = random.choice(range(3))
        else:
            select_idx = self.caption_selection_id

        # 处理pose, 获取body pose的vec表达和6d matrix表达
        global_orient = torch.from_numpy(data_info['global_orient'])
        body_pose = torch.from_numpy(data_info['body_pose'])
        pose = torch.cat([global_orient, body_pose], dim=0)
        body_poseA_rotmat = axis_angle_to_matrix(pose[:66].view(-1, 3))

        task['input'] = random.choice(task['input'])
        task['output'] = random.choice(task['output'])

        item = dict(
            body_poseA_rotmat=body_poseA_rotmat, #  J, 3, 3
            body_poseB_rotmat=torch.zeros_like(body_poseA_rotmat), 
            
            imageA=None, # torch.zeros((3, 336, 336)),  # 按照clip的输入处理
            imageB=None, # torch.zeros((3, 336, 336)), 
            imgA_path=None, 
            imgB_path=None, 
            
            caption=caption_list[select_idx], 
            task=task, 
        )

        return item

class PoseFixDataset(BaseDataset):
    '''
    支持pose-edit任务
    '''
    def __init__(self, 
                 ann_file: str=None, 
                 task_name=None, 
                 **kwargs) -> None:
        super().__init__(ann_file=ann_file, task_name=task_name, **kwargs)
        if task_name is None:
            self.data_infos = []
            return

        data_ids = []
        if self.split == 'train-val':
            for split in ['train', 'val']:
                for seq in ['in', 'out']:
                    with open(osp.join(self.dataset_root, f'{split}_{seq}_sequence_pair_ids.json')) as f:
                        data_ids.extend(json.load(f))
        else:
            for seq in ['in', 'out']:
                with open(osp.join(self.dataset_root, f'{self.split}_{seq}_sequence_pair_ids.json')) as f:
                    data_ids.extend(json.load(f))

        # train or test on human annotated data
        if self.stage == 'finetune':
            with open(osp.join(self.dataset_root, 'posefix_human_6157.json')) as f:
                human_ids = [int(k) for k, _ in json.load(f).items()]
                self.data_infos = [self.data_infos[i] for i in human_ids if i in data_ids]
        else:
            self.data_infos = [self.data_infos[i] for i in data_ids]

    def __len__(self):
        return len(self.data_infos) 

    def __getitem__(self, index):
        data_info = self.data_infos[index]
        task = deepcopy(self.task)
        item = dict()
        
        for pair in ['A', 'B']:
            global_orient = torch.from_numpy(data_info[f'global_orient{pair}'])
            body_pose = torch.from_numpy(data_info[f'body_pose{pair}'])
            pose = torch.cat([global_orient, body_pose], dim=0)
            body_pose_rotmat = axis_angle_to_matrix(pose[:66].view(-1, 3))
            item[f'body_pose{pair}_rotmat'] = body_pose_rotmat # J, 3, 3
            item[f'image{pair}'] = None # torch.zeros_like((3, 336, 336))
            item[f'img{pair}_path'] = ''

        caption_list = data_info['captions']
        if self.split == 'train' and self.stage == 'pretrain':
            select_idx = random.choice(range(3))
        else:
            select_idx = self.caption_selection_id
        task['input'] = random.choice(task['input'])
        task['output'] = random.choice(task['output'])
        item['caption'] = caption_list[select_idx]
        item['task'] = task
        return item

@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        new_batch = dict()
        new_batch['body_poseA_rotmat'] = torch.stack([item['body_poseA_rotmat'] for item in batch])
        new_batch['body_poseB_rotmat'] = torch.stack([item['body_poseB_rotmat'] for item in batch])
        
        if 'keypoints_3d' in batch[0].keys():
            new_batch['keypoints_3d'] = torch.stack([item['keypoints_3d'] for item in batch])
        
        images, hmr_images = [], []
        for item in batch:
            images.append(item['imageA'] if item['imageA'] is not None else torch.zeros(3, 336, 336))
            images.append(item['imageB'] if item['imageB'] is not None else torch.zeros(3, 336, 336))

            hmr_images.append(item['hmr_imageA'] if item.get('hmr_imageA', None) is not None else torch.zeros(3, 256, 256))
            hmr_images.append(item['hmr_imageB'] if item.get('hmr_imageB', None) is not None else torch.zeros(3, 256, 256))
            
        new_batch['images'] = torch.stack(images)
        new_batch['hmr_images'] = torch.stack(hmr_images)
        
        # for vis
        if self.debug:
            new_batch['imgA_path'] = [item['imgA_path'] for item in batch]
            new_batch['imgB_path'] = [item['imgB_path'] for item in batch]

        new_batch['caption'] = [item['caption'] for item in batch]
        new_batch['tasks'] = [item['task'] for item in batch]
        return new_batch

def build_data_module(
        train_dataset_config=None, 
        eval_dataset_config=None, debug=False, **kwargs) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = PoseGPTDataset(**train_dataset_config, **kwargs) if train_dataset_config else None
    eval_dataset = PoseGPTDataset(**eval_dataset_config, **kwargs) if eval_dataset_config else None

    data_collator = DataCollator()
    setattr(data_collator, 'debug', debug)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

# copy from clip image processor

def hmr_transform(n_px):
    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), 
                  (0.229, 0.224, 0.225)),
    ])