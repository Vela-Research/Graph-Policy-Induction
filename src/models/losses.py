"""
Loss functions for founder success prediction.

This module provides loss functions optimized for the imbalanced
classification problem of predicting startup success.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Down-weights easy examples and focuses on hard ones.
    Particularly useful when positive (successful founders) are rare.
    
    Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Raw model outputs (before sigmoid)
            targets: Binary target labels
            
        Returns:
            Focal loss value
        """
        probs = torch.sigmoid(logits)
        
        # p_t = p for y=1, 1-p for y=0
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight: alpha for y=1, (1-alpha) for y=0
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Combined loss
        loss = alpha_weight * focal_weight * bce
        
        return loss.mean()


class PrecisionFocusedLoss(nn.Module):
    """
    Loss function that prioritizes precision over recall.
    
    Uses asymmetric weighting where false positives are penalized
    more heavily than false negatives.
    
    Args:
        fp_weight: Weight for false positive penalties (default 5.0)
    """
    
    def __init__(self, fp_weight: float = 5.0):
        super().__init__()
        self.fp_weight = fp_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute precision-focused loss.
        
        Args:
            logits: Raw model outputs
            targets: Binary target labels
            
        Returns:
            Loss value
        """
        probs = torch.sigmoid(logits)
        
        # False positive loss (predict 1 when actual is 0)
        fp_loss = (1 - targets) * probs * self.fp_weight
        
        # False negative loss (predict 0 when actual is 1)
        fn_loss = targets * (1 - probs)
        
        return (fp_loss + fn_loss).mean()


def get_precision_weighted_loss(
    precision_weight: float = 5.0,
    device: str = 'cpu'
) -> nn.Module:
    """
    Get BCE loss with precision-focused weighting.
    
    This loss makes false positives cost `precision_weight` times
    more than false negatives, which encourages higher precision.
    
    Args:
        precision_weight: FP cost multiplier
        device: Target device
        
    Returns:
        BCEWithLogitsLoss with pos_weight set appropriately
    """
    # pos_weight < 1 penalizes false positives more
    pos_weight = torch.tensor([1.0 / precision_weight]).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_loss_function(
    loss_type: str = 'precision_weighted',
    precision_weight: float = 5.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    device: str = 'cpu'
) -> nn.Module:
    """
    Factory function to get loss function by type.
    
    Args:
        loss_type: Type of loss ('precision_weighted', 'focal', 'bce')
        precision_weight: Weight for precision-focused loss
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        device: Target device
        
    Returns:
        Loss function module
    """
    if loss_type == 'precision_weighted':
        return get_precision_weighted_loss(precision_weight, device)
    elif loss_type == 'focal':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == 'precision_focused':
        return PrecisionFocusedLoss(fp_weight=precision_weight)
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
