# to solve the weight download SSL issue
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
import torchvision.models as models

from ..losses import get_loss
from ..metrics import compute_metrics


__all__ = ['get_model', 'ClassificationModule']


def get_model(model_name, num_classes=2, **kwargs):
    if 'resnet' in model_name:
        model = eval(f'models.{model_name}')(**kwargs)
        model.fc= nn.Linear(in_features=model.fc.in_features, 
                            out_features=num_classes, 
                            bias=True)
        return model
        
    elif 'efficientnet' in model_name:
        model = eval(f'models.{model_name}')(**kwargs)
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, 
                                        out_features=num_classes, 
                                        bias=True)
        return model
    
    elif 'convnext' in model_name:
        model = eval(f'models.{model_name}')(**kwargs)
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, 
                                         out_features=num_classes, 
                                         bias=True)
        return model

    elif 'swin' in model_name:
        model = eval(f'models.{model_name}')(**kwargs)
        model.head = nn.Linear(in_features=model.head.in_features, 
                               out_features=num_classes, 
                               bias=True)
        return model
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class ClassificationModule(LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=0.01, 
                 loss_name='cross_entropy', loss_args={}):
        super().__init__()
        self.model = model
        self.optimizer_args = {'lr': lr, 'weight_decay': weight_decay}
        self.test_ds_prefix = None

        self.test_logits = []
        self.test_label = []
        self.loss_fn = get_loss(loss_name, **loss_args)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        img, label = batch
        logits = self(img)
        loss = self.loss_fn(logits, label)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        if stage == 'train':
            metrics = self.metrics(logits, label)
            for m in metrics:
                self.log(f'{stage}_{m}', metrics[m], prog_bar=True)
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, logits = self._shared_step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        loss, logits = self._shared_step(batch, 'val')
        self.test_logits.append(logits)
        self.test_label.append(label)
        return loss
    
    def on_validation_epoch_end(self):
        metrics = self.metrics(torch.cat(self.test_logits, dim=0), 
                               torch.cat(self.test_label, dim=0))
        for m in metrics:
            self.log(f'val_{m}', metrics[m], on_step=False, on_epoch=True)
        
        self.test_logits.clear()
        self.test_label.clear()
        
    def test_step(self, batch, batch_idx):
        img, label = batch
        loss, logits = self._shared_step(batch, 'test')
        self.test_logits.append(logits)
        self.test_label.append(label)
        return loss

    def on_test_epoch_end(self):
        stage = 'test' if self.test_ds_prefix is None else f'{self.test_ds_prefix}_test'
        metrics = self.metrics(torch.cat(self.test_logits, dim=0), 
                               torch.cat(self.test_label, dim=0))
        for m in metrics:
            self.log(f'{stage}_{m}', metrics[m], on_step=False, on_epoch=True)
        
        self.test_logits.clear()
        self.test_label.clear()

    def configure_optimizers(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        optimizer = AdamW(self.model.parameters(), **self.optimizer_args)
        cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": cosine_scheduler, "interval": "epoch", "frequency": 1}
        }

    def metrics(self, logits, labels):
        pos_score = F.softmax(logits, dim=1)[:, 1].detach()
        challenge_metrics = compute_metrics(labels.cpu().numpy(), pos_score.cpu().numpy())
        return_metrics = {
            "auc": challenge_metrics['AUROC'],
            "auprc": challenge_metrics['AUPRC'],
            "ppv_on_recall90": challenge_metrics['PPV@90% Recall'],
            "accuracy": challenge_metrics['Accuracy'],
            "recall": challenge_metrics['Sensitivity'],
            "specificity": challenge_metrics['Specificity'],
        }
        return return_metrics
    