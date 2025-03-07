import pytorch_lightning as pl
from torch.utils.data import DataLoader
from posegpt.utils.config.config import instantiate_from_config
import torch
import numpy as np

def build_datamodule(cfg):
    return DataModule(dataset_cfg=cfg)

def build_dataset(cfg):
    return instantiate_from_config(cfg)

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_cfg):
        super().__init__()
        self.dataloader_options = {"collate_fn": data_collate}
        self.persistent_workers = True

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self.dataset_cfg = dataset_cfg

        # ================================================
        # for debug
        # ================================================
        debug = False
        if debug:
            self.dataset_cfg.train.samples_per_gpu = 1
            self.dataset_cfg.val.samples_per_gpu = 1
            self.dataset_cfg.test.samples_per_gpu = 1

            self.dataset_cfg.train.workers_per_gpu = 0
            self.dataset_cfg.val.workers_per_gpu = 0
            self.dataset_cfg.test.workers_per_gpu = 0
            self.persistent_workers = False

    def setup(self, stage=None):
        # Use the getter the first time to load the data
        # 每张GPU都会执行该函数, 将dataset设置成property保证数据集仅构造一次
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            train_dataset_cfg = dict(
                target=self.dataset_cfg.target, 
                params=self.dataset_cfg.train
            )
            self._train_dataset = build_dataset(train_dataset_cfg)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            val_dataset_cfg = dict(
                target=self.dataset_cfg.target, 
                params=self.dataset_cfg.val
            )
            self._val_dataset = build_dataset(val_dataset_cfg)
        return self._val_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            test_dataset_cfg = dict(
                target=self.dataset_cfg.target, 
                params=self.dataset_cfg.test
            )
            self._test_dataset = build_dataset(test_dataset_cfg)
        return self._test_dataset

    def train_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.dataset_cfg.train.samples_per_gpu
        dataloader_options["num_workers"] = self.dataset_cfg.train.workers_per_gpu
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            persistent_workers=self.persistent_workers,
            **dataloader_options,
        )
    
    def val_dataloader(self):
        # overrides batch_size and num_workers
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.dataset_cfg.val.samples_per_gpu
        dataloader_options["num_workers"] = self.dataset_cfg.val.workers_per_gpu
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.val_dataset,
            persistent_workers=self.persistent_workers,
            **dataloader_options,
        )
    
    def test_dataloader(self):
        # overrides batch_size and num_workers
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.dataset_cfg.test.samples_per_gpu
        dataloader_options["num_workers"] = self.dataset_cfg.test.workers_per_gpu
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.test_dataset,
            persistent_workers=self.persistent_workers,
            **dataloader_options,
        )

def data_collate(batch):
    new_batch = dict()

    for k, v in batch[0].items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = torch.stack([data[k] for data in batch], dim=0)
        elif isinstance(v, np.ndarray):
            new_batch[k] = torch.stack([torch.tensor(data[k]) for data in batch])
        elif isinstance(v, (dict, str, list)):
            new_batch[k] = [data[k] for data in batch]
        elif isinstance(v, int):
            new_batch[k] = torch.tensor([data[k] for data in batch])
        else:
            raise NotImplementedError

    return new_batch
