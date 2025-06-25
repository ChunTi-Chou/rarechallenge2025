# to solve the weight download SSL issue
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.functional import auroc, recall, precision

import torchvision.models as models


__all__ = ['get_model', 'ClassificationModule']


def compute_ppv_on_recall90(logits, labels):
    """
    Compute PPV (precision) at the threshold that achieves at least 90% recall for the positive class.
    Args:
        logits: torch.Tensor, shape (N, 2), model outputs (logits or probabilities) for each class
        labels: torch.Tensor, shape (N,), ground truth (0 or 1)
    Returns:
        float: PPV at recall >= 90%, or 0.0 if not achievable
    """
    # Use positive class score
    pos_score = F.softmax(logits, dim=1)[:, 1]
    n_pos = labels.sum().item()
    if n_pos == 0:
        return 0.0

    # Sort by score descending
    sorted_idx = torch.argsort(pos_score, descending=True)
    sorted_label = labels[sorted_idx]
    sorted_score = pos_score[sorted_idx]

    # Cumulative TP
    tp_cumsum = torch.cumsum(sorted_label == 1, dim=0)
    recall = tp_cumsum.float() / n_pos

    # Find first threshold where recall >= 0.9
    found = (recall >= 0.9).nonzero(as_tuple=False)
    if len(found) == 0:
        return 0.0
    thresh_idx = found[0].item()
    thresh = sorted_score[thresh_idx]

    # Apply threshold to all samples
    pred_pos = pos_score >= thresh
    tp = ((labels == 1) & pred_pos).sum().item()
    fp = ((labels == 0) & pred_pos).sum().item()
    denom = tp + fp
    if denom == 0:
        return 0.0
    ppv = tp / denom

    return float(ppv)


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
    def __init__(self, model, lr=1e-3, weight_decay=0.01):
        super().__init__()
        self.model = model
        self.optimizer_args = {'lr': lr, 'weight_decay': weight_decay}
        self.test_ds_prefix = None

        self.test_logits = []
        self.test_label = []
        self._init_loss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        img, label = batch
        logits = self(img)
        loss = self.loss(logits, label)
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

    def _init_loss(self):
        self.cn_loss = nn.CrossEntropyLoss()
    
    def loss(self, logits, label):
        return self.cn_loss(logits, label)

    def metrics(self, logits, label):
        auc = auroc(logits, label, task="multiclass", average=None, num_classes=2)[1]
        ppv_on_recall90 = compute_ppv_on_recall90(logits, label)
        
        pred = logits.argmax(dim=1)
        rec = recall(pred, label, task="binary")
        prec = precision(pred, label, task="binary")
        
        return {'auc': auc, 'ppv_on_recall90': ppv_on_recall90, 
                'recall': rec, 'precision': prec}
    