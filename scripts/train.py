import os
import sys
sys.path.append('/home/liyiheng/codes/posegpt')
# nlgmetricverse need this: 
os.environ['MOVERSCORE_MODEL'] = '.model_cache/distilbert-base-uncased'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os.path as osp
import argparse
import time
from posegpt.utils import Config, get_logger
from posegpt.utils.callbacks import build_callback
from posegpt.models import build_modelmodule
from posegpt.datasets import build_datamodule
import pytorch_lightning as pl
import torch.backends.cudnn as cudnn
from posegpt.utils import load_checkpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path', required=True)
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--load-from', help='the checkpoint file to resume from')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    # process config
    cfg = Config.fromfile(args.config)
    cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    cfg.resume_from = None if args.resume_from is None else args.resume_from
    cfg.load_from = None if args.load_from is None else args.load_from

    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # create logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_logger(log_file=log_file)

    # Seed
    pl.seed_everything(args.seed)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # dataset
    datamodule = build_datamodule(cfg.data)

    # model
    model = build_modelmodule(cfg.model, logger)

    # callbacks
    callbacks = build_callback(cfg, logger, phase='train')
    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.work_dir, 
        max_epochs=cfg.max_epochs,
        # precision='32',
        # logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.eval_interval,
        accelerator=cfg.accelerator,
        devices=cfg.device,
        num_nodes=cfg.num_nodes,
        # strategy=DDPStrategy(find_unused_parameters=True) if len(cfg.device) > 1 else 'auto', 
        strategy=DDPStrategy(find_unused_parameters=False) if len(cfg.device) > 1 else 'auto', 
        # strategy='deepspeed_stage_2', 
        # strategy="ddp_find_unused_parameters_true" if len(cfg.device) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
    )

    logger.info("Trainer initialized")

    if cfg.load_from is not None:
        model = load_checkpoint(model, cfg.load_from)
        logger.info(f'load checkpoint from {cfg.load_from}')

    if cfg.resume_from is not None:
        logger.info(f'resume checkpoint from {cfg.resume_from}')

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume_from)

    # Training ends
    logger.info("Training ends!")

if __name__ == '__main__':
    main()

