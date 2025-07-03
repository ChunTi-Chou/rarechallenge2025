from omegaconf import DictConfig, OmegaConf
import hydra

import torch
import torchvision.transforms.v2 as T2
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.models import get_model, ClassificationModule
from src.datasets import Rare25DataModule


def run_experiment(cfg: DictConfig, logger=None):
    # initial settings
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.random_seed)
    if logger is not None:
        logger.log_hyperparams(cfg)
    
    # prepare dataset    
    cfg_preprocessing = cfg.dataset.preprocessing
    preprocessing = None

    cfg_transforms = cfg.dataset.transforms
    train_transform = T2.Compose([
        # Geometric Transformations
        T2.RandomResizedCrop(size=cfg_preprocessing.resize.size, 
                             **cfg_transforms.random_resized_crop),
        T2.RandomRotation(**cfg_transforms.rotate),
        T2.RandomHorizontalFlip(**cfg_transforms.hflip),
        T2.RandomVerticalFlip(**cfg_transforms.vflip),
        T2.ElasticTransform(**cfg_transforms.elastic),
        # Color Transformations
        T2.ColorJitter(**cfg_transforms.colorjitter),
        # Normalization
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(**cfg_preprocessing.normalize)
    ])
    
    test_transform = T2.Compose([
        # Normalization
        T2.Resize(**cfg_preprocessing.resize),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(**cfg_preprocessing.normalize)
    ])

    dm = Rare25DataModule(preprocessing=preprocessing,
                          train_transform=train_transform,
                          test_transform=test_transform,
                          batch_size=cfg.dataset.batch_size,
                          num_workers=cfg.dataset.num_workers,
                          random_seed=cfg.random_seed)
    dm.setup()
    
    # prepare model
    backbone = get_model(cfg.model.model_name, **cfg.model.model_args)
    model = ClassificationModule(backbone, 
                                  lr=cfg.training.optimizer.lr, 
                                  weight_decay=cfg.training.optimizer.weight_decay,
                                  loss_name=cfg.model.loss_name,
                                  loss_args=cfg.model.loss_args)
    
    # prepare training
    checkpoint_callback = ModelCheckpoint(monitor="val_ppv_on_recall90", save_top_k=1, every_n_epochs=1)
    lr_monitor_callback = LearningRateMonitor()

    trainer = Trainer(accelerator=cfg.training.accelerator, 
                      devices=cfg.training.devices,
                      max_epochs=cfg.training.max_epochs, 
                      log_every_n_steps=1,
                      callbacks=[checkpoint_callback, lr_monitor_callback],
                      logger=logger,
                      fast_dev_run=cfg.training.fast_dev_run,
                     )
    
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path='best')


def run_evaluation(cfg: DictConfig, logger=None):
    pass


@hydra.main(version_base=None, config_path='./configs/', config_name='mps_dev_run')
def main(cfg: DictConfig):
    logger = TensorBoardLogger(save_dir=cfg.lightning_log_dir, name=cfg.run_name)
    
    run_experiment(cfg, logger)
    # run_evaluation(cfg, logger)


if __name__ == '__main__':
    main()
