from omegaconf import DictConfig, OmegaConf
import hydra

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from peft import LoraConfig, get_peft_model, TaskType

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
    train_transform = A.Compose([
        # Geometric Transformations
        A.RandomResizedCrop(size=cfg_preprocessing.resize.size, **cfg_transforms.get('random_resized_crop', {})),
        A.GridDistortion(num_steps=5, p=0.5),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=60),
        # Color Transformations
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        # Mask and Blur Transformations
        A.OneOf([
            A.Blur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ], p=0.5),
        A.GridDropout(ratio=0.3, p=0.3),
        # preprocessing
        A.Normalize(**cfg_preprocessing.normalize),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        # preprocessing
        A.Resize(height=cfg_preprocessing.resize.size[0], 
                 width=cfg_preprocessing.resize.size[1]),
        A.Normalize(**cfg_preprocessing.normalize),
        ToTensorV2()
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
    
    # Apply LoRA if enabled in config
    if cfg.model.get('use_lora', False):
        # Determine target modules based on model architecture
        # Only target LoRA-supported layers: Linear, Conv1d/2d/3d, Embedding, MultiheadAttention
        model_name = cfg.model.model_name.lower()
        if 'resnet' in model_name:
            default_target_modules = ['layer1', 'layer2', 'layer3', 'layer4', 'fc']  # Residual layers + classifier
        elif 'efficientnet' in model_name:
            default_target_modules = ['conv', 'classifier']  # Conv layers + classifier
        elif 'convnext' in model_name:
            default_target_modules = ['downsample', 'classifier']  # Downsample layers + classifier
        elif 'swin' in model_name:
            default_target_modules = ['qkv', 'proj', 'mlp.fc1', 'mlp.fc2', 'head']  # Specific Linear layers in Swin
        else:
            # Fallback - only target commonly supported layer types
            default_target_modules = ['conv', 'fc', 'classifier', 'head']
        
        lora_config = LoraConfig(
            # task_type=TaskType.FEATURE_EXTRACTION,  # For image classification backbone
            inference_mode=False,
            r=cfg.model.lora.get('r', 8),  # LoRA rank
            lora_alpha=cfg.model.lora.get('alpha', 32),  # LoRA alpha
            lora_dropout=cfg.model.lora.get('dropout', 0.1),  # LoRA dropout
            target_modules=cfg.model.lora.get('target_modules', default_target_modules),  # Model-specific target modules
        )
        backbone = get_peft_model(backbone, lora_config)
        print(f"LoRA applied to {model_name} with target modules: {lora_config.target_modules}")
        backbone.print_trainable_parameters()
    
    model = ClassificationModule(backbone, 
                                 optimizer_name=cfg.training.optimizer_name,
                                 optimizer_args=cfg.training.optimizer_args,
                                 loss_name=cfg.model.loss_name,
                                 loss_args=cfg.model.loss_args)
    
    # prepare training
    checkpoint_callback = ModelCheckpoint(monitor="val_auprc", mode='max', 
                                          save_top_k=1, every_n_epochs=1)
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
