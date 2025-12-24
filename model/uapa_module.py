"""
UAPA: Uncertainty-Aware Protective Augmentation Module
Implementation based on Section III.B of the paper
"""

import torch
import torch.nn.functional as F
import numpy as np


class UAPAModule:
    """
    Uncertainty-Aware Protective Augmentation (UAPA)
    
    This module implements:
    1. Window-based difficulty computation
    2. Protected region selection
    3. Copy-paste source region selection
    4. Adaptive mixed augmentation
    """
    
    def __init__(self, 
                 window_size=16,          # Window size for dividing feature map
                 beta=0.3,                # Protection ratio
                 rho_max=0.5,            # Maximum paste ratio
                 rho_min=0.1):           # Minimum paste ratio
        """
        Args:
            window_size: Size of each window (feature map will be divided into windows)
            beta: Protection ratio parameter (default: 0.3)
            rho_max: Maximum paste ratio (default: 0.5)
            rho_min: Minimum paste ratio (default: 0.1)
        """
        self.window_size = window_size
        self.beta = beta
        self.rho_max = rho_max
        self.rho_min = rho_min
        
    def compute_window_uncertainty(self, uncertainty_map):
        """
        1. Window-based Difficulty Computation
        Compute uncertainty score for each window: Si,j = (1/|Wi,j|) * Σ U(x,y)
        
        Args:
            uncertainty_map: [B, H, W] uncertainty map
        Returns:
            window_scores: [B, N, N] window uncertainty scores
            N: number of windows in each dimension
        """
        B, H, W = uncertainty_map.shape
        
        # Calculate number of windows
        N_h = H // self.window_size
        N_w = W // self.window_size
        
        # Reshape to windows
        # [B, H, W] -> [B, N_h, window_size, N_w, window_size]
        uncertainty_reshaped = uncertainty_map[:, :N_h*self.window_size, :N_w*self.window_size]
        uncertainty_reshaped = uncertainty_reshaped.reshape(
            B, N_h, self.window_size, N_w, self.window_size
        )
        
        # [B, N_h, window_size, N_w, window_size] -> [B, N_h, N_w, window_size, window_size]
        uncertainty_reshaped = uncertainty_reshaped.permute(0, 1, 3, 2, 4)
        
        # Compute mean uncertainty for each window
        window_scores = uncertainty_reshaped.mean(dim=[3, 4])  # [B, N_h, N_w]
        
        return window_scores, N_h, N_w
    
    def select_protected_regions(self, window_scores, current_epoch, total_epochs):
        """
        2. Protected Region Selection
        Select top-K windows with highest uncertainty as protected regions
        K = ⌊β × (e/E) × N²⌋
        
        Args:
            window_scores: [B, N_h, N_w] window uncertainty scores
            current_epoch: Current training epoch
            total_epochs: Total number of training epochs
        Returns:
            protected_mask: [B, N_h, N_w] binary mask (1 for protected, 0 otherwise)
        """
        B, N_h, N_w = window_scores.shape
        N_total = N_h * N_w
        
        # Compute number of protected regions: K = ⌊β × (e/E) × N²⌋
        K = int(self.beta * (current_epoch / total_epochs) * N_total)
        K = max(1, K)  # At least 1 region
        
        # Flatten window scores
        window_scores_flat = window_scores.reshape(B, -1)  # [B, N_total]
        
        # Get top-K indices
        _, top_k_indices = torch.topk(window_scores_flat, k=K, dim=1)  # [B, K]
        
        # Create protected mask
        protected_mask = torch.zeros_like(window_scores_flat)  # [B, N_total]
        protected_mask.scatter_(1, top_k_indices, 1.0)
        protected_mask = protected_mask.reshape(B, N_h, N_w)  # [B, N_h, N_w]
        
        return protected_mask
    
    def select_copy_paste_sources(self, teacher_pred, protected_mask, 
                                    current_epoch, total_epochs):
        """
        3. Copy-Paste Source Region Selection
        Select windows with high change density from non-protected regions
        
        Args:
            teacher_pred: [B, 2, H, W] teacher predictions (logits)
            protected_mask: [B, N_h, N_w] protected region mask
            current_epoch: Current training epoch
            total_epochs: Total number of training epochs
        Returns:
            paste_mask: [B, N_h, N_w] binary mask indicating regions to paste
        """
        B, _, H, W = teacher_pred.shape
        
        # Get change probability (P1)
        change_prob = F.softmax(teacher_pred, dim=1)[:, 1, :, :]  # [B, H, W]
        
        # Calculate number of windows
        N_h = H // self.window_size
        N_w = W // self.window_size
        
        # Reshape to windows and compute mean change probability per window
        change_prob_reshaped = change_prob[:, :N_h*self.window_size, :N_w*self.window_size]
        change_prob_reshaped = change_prob_reshaped.reshape(
            B, N_h, self.window_size, N_w, self.window_size
        ).permute(0, 1, 3, 2, 4)
        
        change_density = change_prob_reshaped.mean(dim=[3, 4])  # [B, N_h, N_w]
        
        # Mask out protected regions (set to -inf so they won't be selected)
        change_density_masked = change_density.clone()
        change_density_masked[protected_mask > 0] = -float('inf')
        
        # Compute number of paste windows: Nw = ⌊ρ × |A|⌋
        # ρ = ρmax × (1 - e/E) + ρmin
        rho = self.rho_max * (1 - current_epoch / total_epochs) + self.rho_min
        available_regions = (protected_mask == 0).sum(dim=[1, 2])  # [B]
        num_paste = (rho * available_regions).int()  # [B]
        
        # Select top-M windows with highest change density
        paste_mask = torch.zeros_like(change_density)
        for b in range(B):
            M = num_paste[b].item()
            if M > 0:
                change_density_flat = change_density_masked[b].reshape(-1)
                valid_indices = torch.isfinite(change_density_flat)
                if valid_indices.sum() > 0:
                    M = min(M, valid_indices.sum().item())
                    _, top_m_indices = torch.topk(
                        change_density_flat[valid_indices], 
                        k=M
                    )
                    # Map back to original indices
                    original_indices = torch.arange(
                        change_density_flat.size(0), 
                        device=change_density_flat.device
                    )[valid_indices][top_m_indices]
                    
                    paste_mask_flat = paste_mask[b].reshape(-1)
                    paste_mask_flat[original_indices] = 1.0
                    paste_mask[b] = paste_mask_flat.reshape(N_h, N_w)
        
        return paste_mask
    
    def apply_copy_paste(self, images_a, images_b, labels_a, labels_b,
                         protected_mask, paste_mask):
        """
        4. Adaptive Mixed Augmentation
        Apply copy-paste augmentation: Xmix = M ⊙ Xa + (1-M) ⊙ Xb
        
        Args:
            images_a: [B, C, H, W] first set of images
            images_b: [B, C, H, W] second set of images
            labels_a: [B, H, W] first set of labels
            labels_b: [B, H, W] second set of labels
            protected_mask: [B, N_h, N_w] protected region mask
            paste_mask: [B, N_h, N_w] paste region mask
        Returns:
            mixed_images: [B, C, H, W] mixed images
            mixed_labels: [B, H, W] mixed labels
        """
        B, C, H, W = images_a.shape
        
        # Create full-resolution binary mask
        N_h = H // self.window_size
        N_w = W // self.window_size
        
        # Combine protected and paste masks
        # Protected regions: keep original (1)
        # Paste regions: paste from other sample (0)
        # Other regions: random or keep original (1)
        full_mask = torch.ones(B, H, W, device=images_a.device)
        
        for b in range(B):
            for i in range(N_h):
                for j in range(N_w):
                    h_start = i * self.window_size
                    h_end = (i + 1) * self.window_size
                    w_start = j * self.window_size
                    w_end = (j + 1) * self.window_size
                    
                    # If protected, keep as 1 (preserve)
                    if protected_mask[b, i, j] > 0:
                        full_mask[b, h_start:h_end, w_start:w_end] = 1.0
                    # If paste region, set to 0 (copy from b)
                    elif paste_mask[b, i, j] > 0:
                        full_mask[b, h_start:h_end, w_start:w_end] = 0.0
                    # Otherwise keep as 1
        
        # Apply mixing
        full_mask = full_mask.unsqueeze(1)  # [B, 1, H, W]
        mixed_images = full_mask * images_a + (1 - full_mask) * images_b
        
        full_mask = full_mask.squeeze(1)  # [B, H, W]
        mixed_labels = (full_mask * labels_a.float() + 
                       (1 - full_mask) * labels_b.float()).long()
        
        return mixed_images, mixed_labels
    
    def __call__(self, images_a, images_b, labels_a, labels_b,
                 teacher_pred, uncertainty_map, current_epoch, total_epochs):
        """
        Apply complete UAPA augmentation
        
        Args:
            images_a, images_b: [B, C, H, W] unlabeled image pairs
            labels_a, labels_b: [B, H, W] pseudo labels
            teacher_pred: [B, 2, H, W] teacher predictions
            uncertainty_map: [B, H, W] uncertainty map
            current_epoch: Current training epoch
            total_epochs: Total training epochs
        Returns:
            mixed_images: [B, C, H, W] augmented images
            mixed_labels: [B, H, W] augmented labels
        """
        # 1. Compute window-based uncertainty
        window_scores, N_h, N_w = self.compute_window_uncertainty(uncertainty_map)
        
        # 2. Select protected regions
        protected_mask = self.select_protected_regions(
            window_scores, current_epoch, total_epochs
        )
        
        # 3. Select copy-paste source regions
        paste_mask = self.select_copy_paste_sources(
            teacher_pred, protected_mask, current_epoch, total_epochs
        )
        
        # 4. Apply copy-paste augmentation
        mixed_images, mixed_labels = self.apply_copy_paste(
            images_a, images_b, labels_a, labels_b,
            protected_mask, paste_mask
        )
        
        return mixed_images, mixed_labels


if __name__ == "__main__":
    # Test UAPA module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    B, C, H, W = 2, 6, 256, 256
    images_a = torch.randn(B, C, H, W).to(device)
    images_b = torch.randn(B, C, H, W).to(device)
    labels_a = torch.randint(0, 2, (B, H, W)).to(device)
    labels_b = torch.randint(0, 2, (B, H, W)).to(device)
    teacher_pred = torch.randn(B, 2, H, W).to(device)
    uncertainty_map = torch.rand(B, H, W).to(device)
    
    # Initialize UAPA module
    uapa = UAPAModule(window_size=16, beta=0.3, rho_max=0.5, rho_min=0.1)
    
    # Apply UAPA
    mixed_images, mixed_labels = uapa(
        images_a, images_b, labels_a, labels_b,
        teacher_pred, uncertainty_map,
        current_epoch=50, total_epochs=100
    )
    
    print(f"Mixed images shape: {mixed_images.shape}")
    print(f"Mixed labels shape: {mixed_labels.shape}")
    print("UAPA module test passed!")
