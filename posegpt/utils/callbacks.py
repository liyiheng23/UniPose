import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, TQDMProgressBar

class CustomProgressBar(TQDMProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

def build_callback(cfg, logger=None, phase='test'):
    callbacks = []
    # progress bar callback
    callbacks.append(CustomProgressBar())

    # Save 10 latest checkpoints
    if phase == 'train':
        checkpointParams = {
            'dirpath': cfg.work_dir,
            'filename': "{epoch}",
            'monitor': "total_loss",
            'mode': "min",
            'every_n_epochs': 10,
            'save_top_k': 1,
            'save_last': True,
            'save_on_train_epoch_end': True
        }
        callbacks.append(ModelCheckpoint(**checkpointParams))

    return callbacks

