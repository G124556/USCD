"""
DRCL: Difficult Region Contrastive Learning Module
Implementation based on Section III.C of the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DRCLModule(nn.Module):
    """
    Difficult Region Contrastive Learning (DRCL)
    
    This module implements:
    1. Reliable region identification
    2. Local contrastive learning with hard sample mining
    3. Global contrastive learning with prototype memory banks
    """
    
    def __init__(self, 
                 feature_dim=2048,
                 projection_dim=256,
                 num_anchors=32,
                 num_samples=64,
                 temperature=0.1,
                 memory_size=256):
        """
        Args:
            feature_dim: Dimension of input features (2048 for ResNet-50)
            projection_dim: Dimension of projected features
            num_anchors: Number of anchor points (Nr in paper, default: 32)
            num_samples: Number of positive/negative samples per anchor (Ns, default: 64)
            temperature: Temperature parameter for InfoNCE loss (τ, default: 0.1)
            memory_size: Size of prototype memory banks
        """
        super(DRCLModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.num_anchors = num_anchors
        self.num_samples = num_samples
        self.temperature = temperature
        self.memory_size = memory_size
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Conv2d(feature_dim, projection_dim, 1),
            nn.BatchNorm2d(projection_dim),
            nn.ReLU(inplace=True)
        )
        
        # Memory banks for global contrastive learning (FIFO queues)
        self.register_buffer('memory_pos', torch.randn(memory_size, projection_dim))
        self.register_buffer('memory_neg', torch.randn(memory_size, projection_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Normalize memory banks
        self.memory_pos = F.normalize(self.memory_pos, dim=1)
        self.memory_neg = F.normalize(self.memory_neg, dim=1)
    
    def identify_reliable_regions(self, pred_ori, pred_aug):
        """
        Identify reliable regions through prediction consistency
        R(x,y) = I[P_ori_t(x,y) = P_aug_t(x,y)]
        
        Args:
            pred_ori: [B, 2, H, W] predictions on original images
            pred_aug: [B, 2, H, W] predictions on augmented images
        Returns:
            reliable_mask: [B, H, W] binary mask (1 for reliable, 0 otherwise)
        """
        # Get predicted classes
        pred_ori_class = torch.argmax(pred_ori, dim=1)  # [B, H, W]
        pred_aug_class = torch.argmax(pred_aug, dim=1)  # [B, H, W]
        
        # Reliable regions: predictions are consistent
        reliable_mask = (pred_ori_class == pred_aug_class).float()
        
        return reliable_mask
    
    def extract_difficult_region_features(self, features, uncertainty_map, 
                                           reliable_mask, labels, threshold=0.5):
        """
        Extract features from difficult regions (high uncertainty + reliable)
        
        Args:
            features: [B, C, H, W] feature maps
            uncertainty_map: [B, H, W] uncertainty map
            reliable_mask: [B, H, W] reliable region mask
            labels: [B, H, W] ground truth or pseudo labels
            threshold: Uncertainty threshold for difficult regions
        Returns:
            foreground_features: List of foreground feature vectors
            background_features: List of background feature vectors
            foreground_positions: List of (b, h, w) positions
            background_positions: List of (b, h, w) positions
        """
        B, C, H_feat, W_feat = features.shape
        _, H, W = uncertainty_map.shape
        
        # Resize masks to match feature resolution
        if H != H_feat or W != W_feat:
            uncertainty_map = F.interpolate(
                uncertainty_map.unsqueeze(1), 
                size=(H_feat, W_feat), 
                mode='bilinear', 
                align_corners=True
            ).squeeze(1)
            reliable_mask = F.interpolate(
                reliable_mask.unsqueeze(1), 
                size=(H_feat, W_feat), 
                mode='nearest'
            ).squeeze(1)
            labels = F.interpolate(
                labels.unsqueeze(1).float(), 
                size=(H_feat, W_feat), 
                mode='nearest'
            ).squeeze(1).long()
        
        # Project features
        projected_features = self.projection(features)  # [B, projection_dim, H, W]
        
        # Identify difficult regions: high uncertainty + reliable
        difficult_mask = (uncertainty_map > threshold) & (reliable_mask > 0.5)
        
        foreground_features = []
        background_features = []
        foreground_positions = []
        background_positions = []
        
        for b in range(B):
            # Foreground (changed): y = 1
            fg_mask = difficult_mask[b] & (labels[b] == 1)
            if fg_mask.sum() > 0:
                fg_coords = torch.nonzero(fg_mask, as_tuple=False)  # [N, 2]
                for coord in fg_coords:
                    h, w = coord[0].item(), coord[1].item()
                    feat = projected_features[b, :, h, w]
                    foreground_features.append(feat)
                    foreground_positions.append((b, h, w))
            
            # Background (unchanged): y = 0
            bg_mask = difficult_mask[b] & (labels[b] == 0)
            if bg_mask.sum() > 0:
                bg_coords = torch.nonzero(bg_mask, as_tuple=False)  # [N, 2]
                for coord in bg_coords:
                    h, w = coord[0].item(), coord[1].item()
                    feat = projected_features[b, :, h, w]
                    background_features.append(feat)
                    background_positions.append((b, h, w))
        
        return (foreground_features, background_features, 
                foreground_positions, background_positions, 
                projected_features, difficult_mask)
    
    def select_hard_samples(self, anchor, candidates, uncertainty_values, num_samples):
        """
        Select top-N samples with highest uncertainty (hard samples)
        
        Args:
            anchor: [projection_dim] anchor feature
            candidates: List of candidate features
            uncertainty_values: List of uncertainty values for candidates
            num_samples: Number of samples to select
        Returns:
            selected_samples: Tensor of selected features [num_samples, projection_dim]
        """
        if len(candidates) == 0:
            return None
        
        candidates_tensor = torch.stack(candidates)  # [N, projection_dim]
        uncertainty_tensor = torch.tensor(uncertainty_values, device=candidates_tensor.device)
        
        # Select min(num_samples, available samples)
        k = min(num_samples, len(candidates))
        
        if k < len(candidates):
            # Select top-k with highest uncertainty
            _, top_k_indices = torch.topk(uncertainty_tensor, k=k)
            selected_samples = candidates_tensor[top_k_indices]
        else:
            selected_samples = candidates_tensor
        
        return selected_samples
    
    def info_nce_loss(self, query, positive_samples, negative_samples):
        """
        Compute InfoNCE loss
        L = -log[Σ e(q,p+) / (Σ e(q,p+) + Σ e(q,p-))]
        
        Args:
            query: [projection_dim] query feature
            positive_samples: [Np, projection_dim] positive features
            negative_samples: [Nn, projection_dim] negative features
        Returns:
            loss: Scalar loss value
        """
        if positive_samples is None or negative_samples is None:
            return torch.tensor(0.0, device=query.device)
        
        # Normalize features
        query = F.normalize(query, dim=-1)
        positive_samples = F.normalize(positive_samples, dim=-1)
        negative_samples = F.normalize(negative_samples, dim=-1)
        
        # Compute similarity scores
        pos_sim = torch.matmul(positive_samples, query) / self.temperature  # [Np]
        neg_sim = torch.matmul(negative_samples, query) / self.temperature  # [Nn]
        
        # InfoNCE loss
        pos_exp = torch.exp(pos_sim).sum()
        neg_exp = torch.exp(neg_sim).sum()
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
        
        return loss
    
    def local_contrastive_loss(self, foreground_features, background_features,
                                foreground_positions, background_positions,
                                uncertainty_map):
        """
        Compute local contrastive learning loss
        
        Args:
            foreground_features: List of foreground features
            background_features: List of background features
            foreground_positions: List of (b, h, w) positions
            background_positions: List of (b, h, w) positions
            uncertainty_map: [B, H, W] uncertainty map
        Returns:
            loss: Local contrastive loss
        """
        if len(foreground_features) == 0 and len(background_features) == 0:
            return torch.tensor(0.0, device=uncertainty_map.device)
        
        total_loss = 0.0
        num_anchors = 0
        
        # Resize uncertainty map to feature resolution if needed
        _, H, W = uncertainty_map.shape
        
        # Process foreground anchors
        if len(foreground_features) > 0:
            num_fg_anchors = min(self.num_anchors // 2, len(foreground_features))
            anchor_indices = torch.randperm(len(foreground_features))[:num_fg_anchors]
            
            for idx in anchor_indices:
                anchor = foreground_features[idx]
                b, h, w = foreground_positions[idx]
                anchor_uncertainty = uncertainty_map[b, h, w].item()
                
                # Select positive samples (same class - foreground)
                positive_candidates = [f for i, f in enumerate(foreground_features) if i != idx]
                positive_uncertainties = [uncertainty_map[b, h, w].item() 
                                          for i, (b, h, w) in enumerate(foreground_positions) 
                                          if i != idx]
                
                # Select negative samples (different class - background)
                negative_candidates = background_features
                negative_uncertainties = [uncertainty_map[b, h, w].item() 
                                          for (b, h, w) in background_positions]
                
                # Hard sample mining
                positive_samples = self.select_hard_samples(
                    anchor, positive_candidates, positive_uncertainties, self.num_samples
                )
                negative_samples = self.select_hard_samples(
                    anchor, negative_candidates, negative_uncertainties, self.num_samples
                )
                
                # Compute loss
                loss = self.info_nce_loss(anchor, positive_samples, negative_samples)
                total_loss += loss
                num_anchors += 1
        
        # Process background anchors
        if len(background_features) > 0:
            num_bg_anchors = min(self.num_anchors // 2, len(background_features))
            anchor_indices = torch.randperm(len(background_features))[:num_bg_anchors]
            
            for idx in anchor_indices:
                anchor = background_features[idx]
                b, h, w = background_positions[idx]
                anchor_uncertainty = uncertainty_map[b, h, w].item()
                
                # Select positive samples (same class - background)
                positive_candidates = [f for i, f in enumerate(background_features) if i != idx]
                positive_uncertainties = [uncertainty_map[b, h, w].item() 
                                          for i, (b, h, w) in enumerate(background_positions) 
                                          if i != idx]
                
                # Select negative samples (different class - foreground)
                negative_candidates = foreground_features
                negative_uncertainties = [uncertainty_map[b, h, w].item() 
                                          for (b, h, w) in foreground_positions]
                
                # Hard sample mining
                positive_samples = self.select_hard_samples(
                    anchor, positive_candidates, positive_uncertainties, self.num_samples
                )
                negative_samples = self.select_hard_samples(
                    anchor, negative_candidates, negative_uncertainties, self.num_samples
                )
                
                # Compute loss
                loss = self.info_nce_loss(anchor, positive_samples, negative_samples)
                total_loss += loss
                num_anchors += 1
        
        if num_anchors > 0:
            return total_loss / num_anchors
        else:
            return torch.tensor(0.0, device=uncertainty_map.device)
    
    @torch.no_grad()
    def update_memory_bank(self, prototype_pos, prototype_neg):
        """
        Update memory banks with new prototypes (FIFO)
        
        Args:
            prototype_pos: [projection_dim] foreground prototype
            prototype_neg: [projection_dim] background prototype
        """
        ptr = int(self.memory_ptr)
        
        # Normalize prototypes
        prototype_pos = F.normalize(prototype_pos.unsqueeze(0), dim=1)
        prototype_neg = F.normalize(prototype_neg.unsqueeze(0), dim=1)
        
        # Update memory banks
        self.memory_pos[ptr] = prototype_pos
        self.memory_neg[ptr] = prototype_neg
        
        # Update pointer
        ptr = (ptr + 1) % self.memory_size
        self.memory_ptr[0] = ptr
    
    def global_contrastive_loss(self, foreground_features, background_features):
        """
        Compute global contrastive learning loss with prototypes
        
        Args:
            foreground_features: List of foreground features
            background_features: List of background features
        Returns:
            loss: Global contrastive loss
        """
        if len(foreground_features) == 0 or len(background_features) == 0:
            return torch.tensor(0.0, device=self.memory_pos.device)
        
        # Compute current batch prototypes
        fg_prototype = torch.stack(foreground_features).mean(dim=0)  # [projection_dim]
        bg_prototype = torch.stack(background_features).mean(dim=0)  # [projection_dim]
        
        # Normalize
        fg_prototype = F.normalize(fg_prototype, dim=-1)
        bg_prototype = F.normalize(bg_prototype, dim=-1)
        
        # Contrastive loss: foreground prototype vs background memory
        fg_loss = self.info_nce_loss(
            fg_prototype, 
            fg_prototype.unsqueeze(0),  # Self as positive
            self.memory_neg  # Background memory as negative
        )
        
        # Contrastive loss: background prototype vs foreground memory
        bg_loss = self.info_nce_loss(
            bg_prototype,
            bg_prototype.unsqueeze(0),  # Self as positive
            self.memory_pos  # Foreground memory as negative
        )
        
        # Update memory banks
        self.update_memory_bank(fg_prototype, bg_prototype)
        
        return (fg_loss + bg_loss) / 2.0
    
    def forward(self, features, pred_ori, pred_aug, uncertainty_map, 
                reliable_mask, labels):
        """
        Forward pass of DRCL module
        
        Args:
            features: [B, C, H, W] feature maps from encoder
            pred_ori: [B, 2, H, W] predictions on original images
            pred_aug: [B, 2, H, W] predictions on augmented images
            uncertainty_map: [B, H, W] uncertainty map
            reliable_mask: [B, H, W] reliable region mask
            labels: [B, H, W] labels
        Returns:
            loss: Total contrastive loss (Llocal + 0.5*Lglobal)
        """
        # Extract difficult region features
        (fg_features, bg_features, fg_positions, bg_positions, 
         projected_features, difficult_mask) = self.extract_difficult_region_features(
            features, uncertainty_map, reliable_mask, labels
        )
        
        # Local contrastive loss
        local_loss = self.local_contrastive_loss(
            fg_features, bg_features, fg_positions, bg_positions, uncertainty_map
        )
        
        # Global contrastive loss
        global_loss = self.global_contrastive_loss(fg_features, bg_features)
        
        # Total loss: Lcontrast = Llocal + 0.5*Lglobal
        total_loss = local_loss + 0.5 * global_loss
        
        return total_loss


if __name__ == "__main__":
    # Test DRCL module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    B, C, H, W = 2, 2048, 8, 8
    features = torch.randn(B, C, H, W).to(device)
    pred_ori = torch.randn(B, 2, 256, 256).to(device)
    pred_aug = torch.randn(B, 2, 256, 256).to(device)
    uncertainty_map = torch.rand(B, 256, 256).to(device)
    reliable_mask = torch.randint(0, 2, (B, 256, 256)).float().to(device)
    labels = torch.randint(0, 2, (B, 256, 256)).to(device)
    
    # Initialize DRCL module
    drcl = DRCLModule(
        feature_dim=2048,
        projection_dim=256,
        num_anchors=32,
        num_samples=64,
        temperature=0.1,
        memory_size=256
    ).to(device)
    
    # Forward pass
    loss = drcl(features, pred_ori, pred_aug, uncertainty_map, reliable_mask, labels)
    
    print(f"DRCL loss: {loss.item()}")
    print("DRCL module test passed!")
