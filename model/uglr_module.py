"""
UGLR: Uncertainty-Guided Loss Re-weighting Module
Implementation based on Section III.D of the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UGLRModule(nn.Module):
    """
    Uncertainty-Guided Loss Re-weighting (UGLR)
    
    This module implements differentiated loss re-weighting:
    - For labeled data: Higher weights for high-uncertainty regions (γL > 0)
    - For unlabeled data: Lower weights for high-uncertainty regions (γU < 0)
    """
    
    def __init__(self, gamma_labeled=2.0, gamma_unlabeled=-1.0):
        """
        Args:
            gamma_labeled: Weight coefficient for labeled data (default: 2.0)
            gamma_unlabeled: Weight coefficient for unlabeled data (default: -1.0)
        """
        super(UGLRModule, self).__init__()
        
        self.gamma_labeled = gamma_labeled
        self.gamma_unlabeled = gamma_unlabeled
    
    def compute_weights(self, uncertainty_map, is_labeled=True):
        """
        Compute pixel-wise weights based on uncertainty
        
        For labeled data: wL(x,y) = exp(γL × U(x,y))
        For unlabeled data: wU(x,y) = exp(γU × U(x,y))
        
        Args:
            uncertainty_map: [B, H, W] uncertainty map
            is_labeled: Boolean indicating if this is labeled data
        Returns:
            weights: [B, H, W] normalized weights
        """
        if is_labeled:
            # For labeled data: γL > 0, so high uncertainty -> high weight
            weights = torch.exp(self.gamma_labeled * uncertainty_map)
        else:
            # For unlabeled data: γU < 0, so high uncertainty -> low weight
            weights = torch.exp(self.gamma_unlabeled * uncertainty_map)
        
        # Normalize weights (sum to 1 for each sample)
        B = weights.shape[0]
        weights = weights.view(B, -1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        weights = weights.view_as(uncertainty_map)
        
        return weights
    
    def weighted_cross_entropy(self, predictions, targets, weights):
        """
        Compute weighted cross-entropy loss
        
        L = Σ [w(x,y) / Σw(x',y')] × CE(P(x,y), Y(x,y))
        
        Args:
            predictions: [B, C, H, W] predicted logits
            targets: [B, H, W] target labels
            weights: [B, H, W] pixel-wise weights
        Returns:
            loss: Weighted cross-entropy loss
        """
        # Compute cross-entropy loss (no reduction)
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')  # [B, H, W]
        
        # Apply weights
        weighted_loss = ce_loss * weights
        
        # Average over all pixels
        loss = weighted_loss.sum() / (weights.sum() + 1e-8)
        
        return loss
    
    def supervised_loss(self, predictions, targets, uncertainty_map):
        """
        Compute supervised loss with uncertainty-guided weighting
        
        Args:
            predictions: [B, C, H, W] predicted logits
            targets: [B, H, W] ground truth labels
            uncertainty_map: [B, H, W] uncertainty map
        Returns:
            loss: Supervised loss
        """
        # Compute weights for labeled data
        weights = self.compute_weights(uncertainty_map, is_labeled=True)
        
        # Compute weighted loss
        loss = self.weighted_cross_entropy(predictions, targets, weights)
        
        return loss
    
    def unsupervised_loss(self, predictions, pseudo_labels, uncertainty_map,
                          confidence_threshold=0.9):
        """
        Compute unsupervised loss with uncertainty-guided weighting and
        confidence-based filtering
        
        Args:
            predictions: [B, C, H, W] predicted logits
            pseudo_labels: [B, H, W] pseudo labels
            uncertainty_map: [B, H, W] uncertainty map
            confidence_threshold: Threshold for filtering pseudo labels
        Returns:
            loss: Unsupervised loss
        """
        # Compute confidence from teacher predictions
        probs = F.softmax(predictions, dim=1)  # [B, C, H, W]
        confidence, _ = torch.max(probs, dim=1)  # [B, H, W]
        
        # Create confidence mask
        confidence_mask = (confidence > confidence_threshold).float()
        
        # Compute weights for unlabeled data
        weights = self.compute_weights(uncertainty_map, is_labeled=False)
        
        # Apply confidence mask to weights
        weights = weights * confidence_mask
        
        # Avoid division by zero
        if weights.sum() < 1e-8:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute weighted loss
        loss = self.weighted_cross_entropy(predictions, pseudo_labels, weights)
        
        return loss


class TotalLoss(nn.Module):
    """
    Total loss function combining all components:
    Ltotal = Lsup + Lunsup + 0.1*Lcontrast
    """
    
    def __init__(self, 
                 gamma_labeled=2.0, 
                 gamma_unlabeled=-1.0,
                 contrastive_weight=0.1,
                 confidence_threshold=0.9):
        """
        Args:
            gamma_labeled: Weight coefficient for labeled data
            gamma_unlabeled: Weight coefficient for unlabeled data
            contrastive_weight: Weight for contrastive loss (default: 0.1)
            confidence_threshold: Confidence threshold for pseudo labels
        """
        super(TotalLoss, self).__init__()
        
        self.uglr = UGLRModule(
            gamma_labeled=gamma_labeled,
            gamma_unlabeled=gamma_unlabeled
        )
        self.contrastive_weight = contrastive_weight
        self.confidence_threshold = confidence_threshold
    
    def forward(self, 
                student_pred_labeled, labels,
                student_pred_unlabeled, pseudo_labels,
                uncertainty_labeled, uncertainty_unlabeled,
                contrastive_loss=None):
        """
        Compute total loss
        
        Args:
            student_pred_labeled: [B_l, C, H, W] student predictions on labeled data
            labels: [B_l, H, W] ground truth labels
            student_pred_unlabeled: [B_u, C, H, W] student predictions on unlabeled data
            pseudo_labels: [B_u, H, W] pseudo labels
            uncertainty_labeled: [B_l, H, W] uncertainty map for labeled data
            uncertainty_unlabeled: [B_u, H, W] uncertainty map for unlabeled data
            contrastive_loss: Scalar contrastive loss (optional)
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary containing individual loss components
        """
        # Supervised loss
        loss_sup = self.uglr.supervised_loss(
            student_pred_labeled, labels, uncertainty_labeled
        )
        
        # Unsupervised loss
        loss_unsup = self.uglr.unsupervised_loss(
            student_pred_unlabeled, pseudo_labels, uncertainty_unlabeled,
            confidence_threshold=self.confidence_threshold
        )
        
        # Total loss
        total_loss = loss_sup + loss_unsup
        
        loss_dict = {
            'loss_supervised': loss_sup.item(),
            'loss_unsupervised': loss_unsup.item()
        }
        
        # Add contrastive loss if provided
        if contrastive_loss is not None:
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_dict['loss_contrastive'] = contrastive_loss.item()
        
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test UGLR module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data for labeled samples
    B_l, C, H, W = 4, 2, 256, 256
    student_pred_labeled = torch.randn(B_l, C, H, W).to(device)
    labels = torch.randint(0, 2, (B_l, H, W)).to(device)
    uncertainty_labeled = torch.rand(B_l, H, W).to(device)
    
    # Create test data for unlabeled samples
    B_u = 4
    student_pred_unlabeled = torch.randn(B_u, C, H, W).to(device)
    pseudo_labels = torch.randint(0, 2, (B_u, H, W)).to(device)
    uncertainty_unlabeled = torch.rand(B_u, H, W).to(device)
    
    # Initialize loss module
    total_loss_fn = TotalLoss(
        gamma_labeled=2.0,
        gamma_unlabeled=-1.0,
        contrastive_weight=0.1,
        confidence_threshold=0.9
    ).to(device)
    
    # Compute loss
    contrastive_loss = torch.tensor(0.5).to(device)
    total_loss, loss_dict = total_loss_fn(
        student_pred_labeled, labels,
        student_pred_unlabeled, pseudo_labels,
        uncertainty_labeled, uncertainty_unlabeled,
        contrastive_loss
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    print("UGLR module test passed!")
