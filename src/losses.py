import torch
import torch.nn as nn
import torch.nn.functional as F


# the input of the loss function are 
#   - logits of the shape (N, 2)
#   - labels of the shape (N,)

def get_loss(loss_name, **loss_args):
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(**loss_args)
    if loss_name == 'focal_loss':
        return FocalLoss(**loss_args)
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")


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

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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

    