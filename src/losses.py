import torch
import torch.nn as nn
import torch.nn.functional as F


# the input of the loss function are 
#   - logits of the shape (N, 2)
#   - labels of the shape (N,)

def get_loss(loss_name, **loss_args):
    if loss_name == 'cross_entropy':
        return CrossEntropyLoss(**loss_args)
    if loss_name == 'focal_loss':
        return FocalLoss(**loss_args)
    elif loss_name == 'contropy_loss':
        return ContropyLoss(**loss_args)
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, labels)

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Args:
        alpha (float): balancing factor for the classes (0 < alpha < 1)
        gamma (float): focusing parameter for modulating factor (1 - p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (N, 2) - raw, unnormalized scores for each class
            labels: Tensor of shape (N,) - ground truth class indices (0 or 1)
        Returns:
            Focal loss value (scalar tensor)
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = softmax probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ContropyLoss(nn.Module):
    def __init__(self, margin=0.0, alpha=0.1):
        super(ContropyLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        cn_loss = F.cross_entropy(logits, labels)

        # compute the feature loss
        # Step 1: Create all pairwise indices (excluding self-pairs)
        batch_size = features.shape[0]
        # Get all pairs (i, j) where i != j
        i_idx = torch.arange(batch_size).repeat_interleave(batch_size-1)
        j_idx = torch.cat([torch.cat([torch.arange(i), torch.arange(i+1, batch_size)]) for i in range(batch_size)])
        # Step 2: Gather the pairs
        x1 = features[i_idx]   # [310*309, 768]
        x2 = features[j_idx]   # [310*309, 768]
        # Step 3: Create the target tensor (for example, all 1s if you want to maximize similarity)
        target = -1 * torch.ones_like(i_idx, dtype=torch.float).to(features.device)
        contrastive_loss = F.cosine_embedding_loss(x1, x2, target)

        return cn_loss + self.alpha * contrastive_loss