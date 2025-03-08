import pytorch_lightning as pl
from posegpt.utils.config.config import instantiate_from_config, get_obj_from_str
from collections import defaultdict
import os.path as osp
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch

def build_model(cfg):
    return instantiate_from_config(cfg)

def build_modelmodule(cfg, logger=None):
    if not "target" in cfg:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(cfg["target"])(
        **cfg.get("params", dict()), 

        logger=logger, 
        loss_config=cfg.loss_config, 
        metric_config=cfg.metric_config, 
        optimizer_config=cfg.optimizer_config, 
        lr_config=cfg.lr_config, 
        checkpoint_config=cfg.checkpoint_config, 
    )

class BaseModule(pl.LightningModule):
    def __init__(self, 
                 logger=None, 
                 loss_config=None, 
                 metric_config=None, 
                 optimizer_config=None, 
                 lr_config=None, 
                 checkpoint_config=None, 
                 **kwargs) -> None:
        super().__init__()
        if logger is not None:
            self.metric_writter = MetricLogUtil(logger)
        if checkpoint_config is not None:
            self.ckp_writter = CheckpointSaveUtil(logger, **checkpoint_config)

        if loss_config is not None:
            self.loss = build_model(loss_config)

        if metric_config is not None:
            self.metric = build_model(metric_config)
        
        if optimizer_config is not None:
            self.optimizer_config = optimizer_config
        
        if lr_config is not None:
            self.lr_config = lr_config

    def configure_optimizers(self):
        if self.optimizer_config is None:
            return {}

        # Optimizer
        optim_target = self.optimizer_config.target
        if len(optim_target.split('.')) == 1:
            optim_target = 'torch.optim.' + optim_target
        optimizer = get_obj_from_str(optim_target)(
            params=self.parameters(), **self.optimizer_config.params)

        # Scheduler
        scheduler_target = self.lr_config.target
        if len(scheduler_target.split('.')) == 1:
            scheduler_target = 'torch.optim.lr_scheduler.' + scheduler_target
        lr_scheduler = get_obj_from_str(scheduler_target)(
            optimizer=optimizer, **self.lr_config.params)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def on_train_epoch_end(self) -> None:
        losses = self.loss.compute()
        total_loss = sum([v for v in losses.values()])
        losses.update({'total_loss': total_loss})
        self.log('total_loss', total_loss)
        self.metric_writter.log(losses, self.trainer, stage='Train')
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log('learning_rate', self.trainer.optimizers[0].state_dict()['param_groups'][0]['lr'], logger=True)

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            metric = self.metric.compute()
            self.metric_writter.log(metric, self.trainer, stage='Val')
            self.ckp_writter.save_checkpoint(metric, self.trainer)

    @torch.no_grad()
    def on_test_epoch_end(self) -> None:
        metric = self.metric.compute()
        self.log_dict(metric, sync_dist=True, rank_zero_only=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

class MetricLogUtil:
    def __init__(self, logger):
        self.logger = logger
        self.precision = 4
    
    @rank_zero_only
    def log(self, log_dict, trainer, stage='Train'):
        line = f'{stage} - Epoch: {trainer.current_epoch}: '
        log_line = [f'{k}: {round(v.clone().detach().item(), self.precision)}' for k, v in log_dict.items()]
        line = line + ' '.join(log_line)
        self.logger.info(line)

class CheckpointSaveUtil:
    def __init__(self, logger, monitor_variable_name='', monitor_variable_mode='max'):
        self.logger = logger
        assert monitor_variable_mode in ['min', 'max']
        self.monitor_variable_name = monitor_variable_name
        self.monitor_variable_mode = monitor_variable_mode
        if monitor_variable_mode == 'max':
            self.monitor_variable_value = -1e6
        else:
            self.monitor_variable_value = 1e6

    @rank_zero_only
    def save_checkpoint(self, metric_dict, trainer):
        # save ckp by interval
        assert self.monitor_variable_name in metric_dict
        epoch = trainer.current_epoch
        # save best ckp
        if self.monitor_variable_mode == 'max':
            if metric_dict[self.monitor_variable_name] > self.monitor_variable_value:
                self.monitor_variable_value = metric_dict[self.monitor_variable_name]
                save_path = osp.join(trainer._default_root_dir, f'best_{self.monitor_variable_name}.ckpt')
                trainer.save_checkpoint(save_path, weights_only=False)
                self.logger.info(f'Save checkpoint of epoch {epoch} with best {self.monitor_variable_name} to {save_path}')
        else:
            if metric_dict[self.monitor_variable_name] < self.monitor_variable_value:
                self.monitor_variable_value = metric_dict[self.monitor_variable_name]
                save_path = osp.join(trainer._default_root_dir, f'best_{self.monitor_variable_name}.ckpt')
                trainer.save_checkpoint(save_path, weights_only=False)
                self.logger.info(f'Save checkpoint of epoch {epoch} with best {self.monitor_variable_name} to {save_path}')
