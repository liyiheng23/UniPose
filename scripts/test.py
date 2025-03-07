import os
import sys
sys.path.append('/home/liyiheng/codes/posegpt')
os.environ['MOVERSCORE_MODEL'] = 'dataset_preprocess/distilbert-base-uncased'
import argparse
from posegpt.utils import Config
from posegpt.utils.callbacks import build_callback
from posegpt.models import build_modelmodule
from posegpt.datasets import build_datamodule
import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path', required=True)
    parser.add_argument('--checkpoints', help='checkpoint file path', required=True)
    # parser.add_argument('--work-dir', help='the dir to save logs and models')
    # parser.add_argument('--resume-from', help='the checkpoint file to resume from')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # cfg.work_dir = args.work_dir
    # cfg.resume_from = args.resume_from

    # Seed
    pl.seed_everything(args.seed)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # dataset
    datamodule = build_datamodule(cfg.data)

    # model
    model = build_modelmodule(cfg.model)

    # callbacks
    callbacks = build_callback(cfg, None, phase='test')

    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.work_dir, 
        max_epochs=cfg.max_epochs,
        # precision='16',
        # logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.eval_interval,
        accelerator=cfg.accelerator,
        devices=cfg.device,
        num_nodes=cfg.num_nodes,
        # strategy="ddp_find_unused_parameters_true" if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
    )

    trainer.test(model, datamodule=datamodule, ckpt_path=args.checkpoints)[0]

if __name__ == '__main__':
    main()