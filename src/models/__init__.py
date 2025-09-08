# to solve the weight download SSL issue
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from lightning import LightningModule
import torchvision.models as models

from ..losses import get_loss
from ..metrics import compute_metrics


__all__ = ['get_model', 'ClassificationModule']

# SAM optimizer implementation
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


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

        # register forward hook to collect features
        model.features_buffer = []
        model._feature_hook_handle = model.flatten.register_forward_hook(get_hook_fn(model))
        return model
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_hook_fn(model):
    def hook_fn(module, input, output):
        model.features_buffer.append(output)
    return hook_fn


class ClassificationModule(LightningModule):
    def __init__(self, model, 
                 optimizer_name='AdamW', optimizer_args={'lr': 1e-3, 'weight_decay': 0.01}, 
                 loss_name='cross_entropy', loss_args={}):
        super().__init__()
        self.model = model
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        self.test_ds_prefix = None

        self.test_logits = []
        self.test_label = []
        self.loss_fn = get_loss(loss_name, **loss_args)
        
        # Disable automatic optimization for SAM
        if optimizer_name == 'SAM':
            self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        img, label = batch
        # compute the classification loss
        logits = self(img)
        features = torch.cat(self.model.features_buffer, dim=0)
        loss = self.loss_fn(logits, label, features)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.model.features_buffer.clear()

        if stage == 'train':
            metrics = self.metrics(logits, label)
            for m in metrics:
                self.log(f'{stage}_{m}', metrics[m], prog_bar=True)
        return loss, logits

    def training_step(self, batch, batch_idx):
        if self.optimizer_name == 'SAM':
            # First forward-backward pass for SAM
            optimizer = self.optimizers()
            optimizer.zero_grad()
            loss, logits = self._shared_step(batch, 'train')
            self.manual_backward(loss)
            optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass for SAM
            loss, logits = self._shared_step(batch, 'train')
            self.manual_backward(loss)
            optimizer.second_step(zero_grad=True)
            
            return loss
        else:
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
        if self.optimizer_name == 'AdamW':
            optimizer = AdamW(self.model.parameters(), **self.optimizer_args)
            cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-6)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": cosine_scheduler, "interval": "epoch", "frequency": 1}
            }
        elif self.optimizer_name == 'SAM':
            base_optimizer = SGD
            optimizer = SAM(self.model.parameters(), base_optimizer, **self.optimizer_args)
            return optimizer


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
    