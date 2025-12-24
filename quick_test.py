"""
Quick test script to verify USCD implementation
Tests all components without requiring a full dataset
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

print("="*60)
print("USCD Implementation Test")
print("="*60)

# Test device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Test imports
print("\n1. Testing imports...")
try:
    from model.uscd_model import USCDModel
    from model.uapa_module import UAPAModule
    from model.drcl_module import DRCLModule
    from model.uglr_module import TotalLoss
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test model creation
print("\n2. Testing model creation...")
try:
    model = USCDModel(num_classes=2, pretrained=False).to(device)
    print(f"   ✓ Model created successfully")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    sys.exit(1)

# Test forward pass
print("\n3. Testing forward pass...")
try:
    B, C, H, W = 2, 3, 256, 256
    img_a = torch.randn(B, C, H, W).to(device)
    img_b = torch.randn(B, C, H, W).to(device)
    
    # Student forward
    output_s, features_s = model(img_a, img_b, is_student=True)
    print(f"   ✓ Student forward pass successful")
    print(f"   - Output shape: {output_s.shape}")
    print(f"   - Features shape: {features_s.shape}")
    
    # Teacher forward
    output_t, features_t = model(img_a, img_b, is_student=False)
    print(f"   ✓ Teacher forward pass successful")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test uncertainty computation
print("\n4. Testing uncertainty computation...")
try:
    uncertainty = model.compute_uncertainty(output_t)
    print(f"   ✓ Uncertainty computation successful")
    print(f"   - Uncertainty shape: {uncertainty.shape}")
    print(f"   - Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
except Exception as e:
    print(f"   ✗ Uncertainty computation failed: {e}")
    sys.exit(1)

# Test EMA update
print("\n5. Testing EMA update...")
try:
    model.update_teacher(momentum=0.999)
    print(f"   ✓ Teacher EMA update successful")
except Exception as e:
    print(f"   ✗ EMA update failed: {e}")
    sys.exit(1)

# Test UAPA module
print("\n6. Testing UAPA module...")
try:
    uapa = UAPAModule(window_size=16, beta=0.3, rho_max=0.5, rho_min=0.1)
    
    imgs_a = torch.cat([img_a, img_b], dim=1)  # [B, 6, H, W]
    imgs_b = torch.cat([img_a, img_b], dim=1)
    labels_a = torch.randint(0, 2, (B, H, W)).to(device)
    labels_b = torch.randint(0, 2, (B, H, W)).to(device)
    
    mixed_imgs, mixed_labels = uapa(
        imgs_a, imgs_b, labels_a, labels_b,
        output_t, uncertainty,
        current_epoch=50, total_epochs=100
    )
    
    print(f"   ✓ UAPA module successful")
    print(f"   - Mixed images shape: {mixed_imgs.shape}")
    print(f"   - Mixed labels shape: {mixed_labels.shape}")
except Exception as e:
    print(f"   ✗ UAPA module failed: {e}")
    sys.exit(1)

# Test DRCL module
print("\n7. Testing DRCL module...")
try:
    drcl = DRCLModule(
        feature_dim=2048,
        projection_dim=256,
        num_anchors=32,
        num_samples=64,
        temperature=0.1,
        memory_size=256
    ).to(device)
    
    labels = torch.randint(0, 2, (B, H, W)).to(device)
    reliable_mask = torch.randint(0, 2, (B, H, W)).float().to(device)
    
    contrastive_loss = drcl(
        features_s, output_t, output_t,
        uncertainty, reliable_mask, labels
    )
    
    print(f"   ✓ DRCL module successful")
    print(f"   - Contrastive loss: {contrastive_loss.item():.4f}")
except Exception as e:
    print(f"   ✗ DRCL module failed: {e}")
    sys.exit(1)

# Test UGLR module
print("\n8. Testing UGLR module...")
try:
    total_loss_fn = TotalLoss(
        gamma_labeled=2.0,
        gamma_unlabeled=-1.0,
        contrastive_weight=0.1,
        confidence_threshold=0.9
    ).to(device)
    
    # Create dummy data
    student_pred_l = torch.randn(B, 2, H, W).to(device)
    student_pred_u = torch.randn(B, 2, H, W).to(device)
    labels_l = torch.randint(0, 2, (B, H, W)).to(device)
    pseudo_labels_u = torch.randint(0, 2, (B, H, W)).to(device)
    uncertainty_l = torch.rand(B, H, W).to(device)
    uncertainty_u = torch.rand(B, H, W).to(device)
    
    total_loss, loss_dict = total_loss_fn(
        student_pred_l, labels_l,
        student_pred_u, pseudo_labels_u,
        uncertainty_l, uncertainty_u,
        contrastive_loss
    )
    
    print(f"   ✓ UGLR module successful")
    print(f"   - Total loss: {total_loss.item():.4f}")
    print(f"   - Loss components:")
    for key, value in loss_dict.items():
        print(f"     • {key}: {value:.4f}")
except Exception as e:
    print(f"   ✗ UGLR module failed: {e}")
    sys.exit(1)

# Test backward pass
print("\n9. Testing backward pass...")
try:
    optimizer = torch.optim.SGD(model.student.parameters(), lr=0.01)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(f"   ✓ Backward pass and optimization successful")
except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")
    sys.exit(1)

# Test gradient clipping
print("\n10. Testing gradient clipping...")
try:
    torch.nn.utils.clip_grad_norm_(model.student.parameters(), max_norm=1.0)
    print(f"   ✓ Gradient clipping successful")
except Exception as e:
    print(f"   ✗ Gradient clipping failed: {e}")
    sys.exit(1)

# Memory usage
if torch.cuda.is_available():
    print("\n11. GPU Memory Usage:")
    print(f"   - Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    print(f"   - Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nNext steps:")
print("1. Prepare your dataset following the structure in README.md")
print("2. Run training: python train.py --data_root /path/to/dataset --label_ratio 0.05")
print("3. Run testing: python test.py --data_root /path/to/dataset --checkpoint ./checkpoints/best.pth")
print("\nFor detailed usage, see README.md")
print("="*60 + "\n")
