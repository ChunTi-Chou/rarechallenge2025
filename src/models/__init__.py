import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.functional import auroc, recall, precision


from .swin_v2 import SwinTransformerV2


__all__ = ['get_model', 'ClassificationModule']


def get_model(model_name, **kwargs):
    if model_name == 'swin_v2':
        return SwinTransformerV2(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class ClassificationModule(LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=0.01):
        super().__init__()
        self.model = model
        self.optimizer_args = {'lr': lr, 'weight_decay': weight_decay}
        self.test_ds_prefix = None

        self.test_pred = []
        self.test_label = []

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        img, label = batch
        pred = self(img)
        loss = self.loss(pred, label)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        if stage != 'test':
            metrics = self.metrics(pred, label)
            for m in metrics:
                self.log(f'{stage}_{m}', metrics[m], prog_bar=True)
        return loss, pred

    def training_step(self, batch, batch_idx):
        loss, pred = self._shared_step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred = self._shared_step(batch, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        loss, pred = self._shared_step(batch, 'test')
        self.test_pred.append(pred)
        self.test_label.append(label)
        return loss

    def on_test_epoch_end(self):
        stage = 'test' if self.test_ds_prefix is None else f'{self.test_ds_prefix}_test'
        metrics = self.metrics(torch.cat(self.test_pred, dim=0), 
                               torch.cat(self.test_label, dim=0))
        for m in metrics:
            self.log(f'{stage}_{m}', metrics[m], on_step=False, on_epoch=True)
        
        self.test_pred.clear()
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

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)

    def metrics(self, pred, label):
        auc = auroc(pred, label, task="multiclass", average=None, num_classes=2)[1]
        rec = recall(pred.argmax(dim=1), label, task="binary")
        prec = precision(pred.argmax(dim=1), label, task="binary")
        return {'auc': auc, 'recall': rec, 'precision': prec}
    