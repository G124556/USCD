"""
USCD Training Script
Complete implementation of uncertainty-guided semi-supervised change detection
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse

from model.uscd_model import USCDModel
from model.uapa_module import UAPAModule
from model.drcl_module import DRCLModule
from model.uglr_module import TotalLoss
from dataset import create_dataloaders


class USCDTrainer:
    """
    Complete USCD Training Framework
    Implements Algorithm 1 from the paper
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        print("Creating USCD model...")
        self.model = USCDModel(
            num_classes=args.num_classes,
            pretrained=args.pretrained
        ).to(self.device)
        
        # Create modules
        print("Initializing USCD modules...")
        self.uapa = UAPAModule(
            window_size=args.window_size,
            beta=args.beta,
            rho_max=args.rho_max,
            rho_min=args.rho_min
        )
        
        self.drcl = DRCLModule(
            feature_dim=2048,
            projection_dim=256,
            num_anchors=args.num_anchors,
            num_samples=args.num_samples,
            temperature=args.temperature,
            memory_size=args.memory_size
        ).to(self.device)
        
        self.criterion = TotalLoss(
            gamma_labeled=args.gamma_labeled,
            gamma_unlabeled=args.gamma_unlabeled,
            contrastive_weight=args.contrastive_weight,
            confidence_threshold=args.confidence_threshold
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.SGD(
            list(self.model.student.parameters()) + list(self.drcl.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=args.lr_end_factor,
            total_iters=args.epochs
        )
        
        # Create dataloaders
        print(f"Loading dataset from {args.data_root}...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_ratio=args.label_ratio,
            image_size=args.image_size
        )
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=args.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_f1 = 0.0
        
        print(f"Training on device: {self.device}")
        print(f"Total epochs: {args.epochs}")
        print(f"Supervised warmup epochs: {args.warmup_epochs}")
    
    def train_epoch(self, epoch):
        """
        Train one epoch
        """
        self.model.train()
        self.drcl.train()
        
        epoch_losses = {'total': [], 'sup': [], 'unsup': [], 'contrast': []}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Get data
            img_a = batch['img_a'].to(self.device)
            img_b = batch['img_b'].to(self.device)
            labels = batch['label'].to(self.device)
            is_labeled = batch['is_labeled']
            
            # Separate labeled and unlabeled data
            labeled_mask = is_labeled.to(self.device)
            labeled_indices = torch.where(labeled_mask)[0]
            unlabeled_indices = torch.where(~labeled_mask)[0]
            
            if len(labeled_indices) == 0:
                continue
            
            # === Supervised Warmup (first 30 epochs) ===
            if epoch < self.args.warmup_epochs:
                # Only use labeled data
                img_a_l = img_a[labeled_indices]
                img_b_l = img_b[labeled_indices]
                labels_l = labels[labeled_indices]
                
                # Student forward
                student_pred, _ = self.model(img_a_l, img_b_l, is_student=True)
                
                # Teacher forward (for uncertainty)
                with torch.no_grad():
                    teacher_pred, _ = self.model(img_a_l, img_b_l, is_student=False)
                    uncertainty = self.model.compute_uncertainty(teacher_pred)
                
                # Supervised loss only
                loss = self.criterion.uglr.supervised_loss(
                    student_pred, labels_l, uncertainty
                )
                
                loss_dict = {
                    'loss_supervised': loss.item(),
                    'loss_unsupervised': 0.0,
                    'loss_contrastive': 0.0,
                    'loss_total': loss.item()
                }
            
            # === Semi-supervised Training (after warmup) ===
            else:
                # Process labeled data
                img_a_l = img_a[labeled_indices]
                img_b_l = img_b[labeled_indices]
                labels_l = labels[labeled_indices]
                
                # Student forward on labeled
                student_pred_l, features_l = self.model(img_a_l, img_b_l, is_student=True)
                
                # Teacher forward on labeled
                with torch.no_grad():
                    teacher_pred_l, _ = self.model(img_a_l, img_b_l, is_student=False)
                    uncertainty_l = self.model.compute_uncertainty(teacher_pred_l)
                
                # Process unlabeled data if available
                if len(unlabeled_indices) > 0:
                    img_a_u = img_a[unlabeled_indices]
                    img_b_u = img_b[unlabeled_indices]
                    
                    # Teacher predictions for pseudo labels and uncertainty
                    with torch.no_grad():
                        teacher_pred_u, _ = self.model(img_a_u, img_b_u, is_student=False)
                        uncertainty_u = self.model.compute_uncertainty(teacher_pred_u)
                        pseudo_labels_u = torch.argmax(teacher_pred_u, dim=1)
                        
                        # UAPA: Generate augmented data
                        # For simplicity, we use different samples within batch
                        if len(unlabeled_indices) >= 2:
                            half = len(unlabeled_indices) // 2
                            img_a_u1 = img_a_u[:half]
                            img_b_u1 = img_b_u[:half]
                            img_a_u2 = img_a_u[half:2*half]
                            img_b_u2 = img_b_u[half:2*half]
                            pseudo_u1 = pseudo_labels_u[:half]
                            pseudo_u2 = pseudo_labels_u[half:2*half]
                            teacher_pred_u_crop = teacher_pred_u[:half]
                            uncertainty_u_crop = uncertainty_u[:half]
                            
                            # Concatenate img_a and img_b for augmentation
                            imgs_1 = torch.cat([img_a_u1, img_b_u1], dim=1)
                            imgs_2 = torch.cat([img_a_u2, img_b_u2], dim=1)
                            
                            # Apply UAPA
                            mixed_imgs, mixed_labels_u = self.uapa(
                                imgs_1, imgs_2, pseudo_u1, pseudo_u2,
                                teacher_pred_u_crop, uncertainty_u_crop,
                                epoch, self.args.epochs
                            )
                            
                            # Split back to img_a and img_b
                            mixed_img_a_u = mixed_imgs[:, :3, :, :]
                            mixed_img_b_u = mixed_imgs[:, 3:, :, :]
                            
                            # Student forward on augmented unlabeled
                            student_pred_u, features_u = self.model(
                                mixed_img_a_u, mixed_img_b_u, is_student=True
                            )
                            
                            # Teacher forward on augmented for reliable mask
                            with torch.no_grad():
                                teacher_pred_u_aug, _ = self.model(
                                    mixed_img_a_u, mixed_img_b_u, is_student=False
                                )
                                uncertainty_u_final = self.model.compute_uncertainty(teacher_pred_u_aug)
                        else:
                            # Not enough samples for UAPA
                            student_pred_u, features_u = self.model(img_a_u, img_b_u, is_student=True)
                            mixed_labels_u = pseudo_labels_u
                            uncertainty_u_final = uncertainty_u
                            teacher_pred_u_aug = teacher_pred_u
                    
                    # Compute DRCL loss
                    # For labeled data
                    with torch.no_grad():
                        teacher_pred_l_ori, _ = self.model(img_a_l, img_b_l, is_student=False)
                        teacher_pred_l_aug = teacher_pred_l  # Use same for simplicity
                        reliable_mask_l = self.drcl.identify_reliable_regions(
                            teacher_pred_l_ori, teacher_pred_l_aug
                        )
                    
                    contrastive_loss = self.drcl(
                        features_l, teacher_pred_l_ori, teacher_pred_l_aug,
                        uncertainty_l, reliable_mask_l, labels_l
                    )
                    
                    # Total loss
                    loss, loss_dict = self.criterion(
                        student_pred_l, labels_l,
                        student_pred_u, mixed_labels_u,
                        uncertainty_l, uncertainty_u_final,
                        contrastive_loss
                    )
                else:
                    # Only labeled data available
                    loss = self.criterion.uglr.supervised_loss(
                        student_pred_l, labels_l, uncertainty_l
                    )
                    loss_dict = {
                        'loss_supervised': loss.item(),
                        'loss_unsupervised': 0.0,
                        'loss_contrastive': 0.0,
                        'loss_total': loss.item()
                    }
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.student.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update teacher with EMA
            self.model.update_teacher(momentum=self.args.ema_momentum)
            
            # Record losses
            for key in ['total', 'sup', 'unsup', 'contrast']:
                loss_key = f'loss_{key}' if key != 'total' else 'loss_total'
                if key == 'sup':
                    loss_key = 'loss_supervised'
                elif key == 'unsup':
                    loss_key = 'loss_unsupervised'
                elif key == 'contrast':
                    loss_key = 'loss_contrastive'
                
                if loss_key in loss_dict:
                    epoch_losses[key].append(loss_dict[loss_key])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss_total']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Average epoch losses
        avg_losses = {k: np.mean(v) if len(v) > 0 else 0.0 
                      for k, v in epoch_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self):
        """
        Validate the model
        """
        self.model.eval()
        
        tp, fp, fn, tn = 0, 0, 0, 0
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            img_a = batch['img_a'].to(self.device)
            img_b = batch['img_b'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Student prediction
            pred, _ = self.model(img_a, img_b, is_student=True)
            pred_class = torch.argmax(pred, dim=1)
            
            # Compute metrics
            tp += ((pred_class == 1) & (labels == 1)).sum().item()
            fp += ((pred_class == 1) & (labels == 0)).sum().item()
            fn += ((pred_class == 0) & (labels == 1)).sum().item()
            tn += ((pred_class == 0) & (labels == 0)).sum().item()
        
        # Compute metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'drcl_state_dict': self.drcl.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_f1': self.best_f1
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.args.save_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.args.save_dir, 'best.pth'))
            print(f"Saved best model with F1: {metrics['f1']:.4f}")
    
    def train(self):
        """
        Main training loop
        """
        print("\n" + "="*50)
        print("Starting USCD Training")
        print("="*50 + "\n")
        
        for epoch in range(1, self.args.epochs + 1):
            self.current_epoch = epoch
            
            # Train one epoch
            train_losses = self.train_epoch(epoch)
            
            # Log training losses
            for key, value in train_losses.items():
                self.writer.add_scalar(f'Train/loss_{key}', value, epoch)
            
            # Validate
            if epoch % self.args.val_interval == 0:
                val_metrics = self.validate()
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'Val/{key}', value, epoch)
                
                print(f"\nEpoch {epoch} - Val F1: {val_metrics['f1']:.4f}, "
                      f"IoU: {val_metrics['iou']:.4f}, "
                      f"Precision: {val_metrics['precision']:.4f}, "
                      f"Recall: {val_metrics['recall']:.4f}\n")
                
                # Save checkpoint
                is_best = val_metrics['f1'] > self.best_f1
                if is_best:
                    self.best_f1 = val_metrics['f1']
                
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Update learning rate
            self.scheduler.step()
        
        print("\n" + "="*50)
        print(f"Training completed! Best F1: {self.best_f1:.4f}")
        print("="*50 + "\n")
        
        # Final test
        print("Running final test...")
        test_metrics = self.test()
        print(f"Test F1: {test_metrics['f1']:.4f}, IoU: {test_metrics['iou']:.4f}")
        
        self.writer.close()
    
    @torch.no_grad()
    def test(self):
        """
        Test the model
        """
        # Load best model
        checkpoint = torch.load(os.path.join(self.args.save_dir, 'best.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        tp, fp, fn, tn = 0, 0, 0, 0
        
        for batch in tqdm(self.test_loader, desc='Testing'):
            img_a = batch['img_a'].to(self.device)
            img_b = batch['img_b'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Student prediction
            pred, _ = self.model(img_a, img_b, is_student=True)
            pred_class = torch.argmax(pred, dim=1)
            
            # Compute metrics
            tp += ((pred_class == 1) & (labels == 1)).sum().item()
            fp += ((pred_class == 1) & (labels == 0)).sum().item()
            fn += ((pred_class == 0) & (labels == 1)).sum().item()
            tn += ((pred_class == 0) & (labels == 0)).sum().item()
        
        # Compute metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy
        }
        
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='USCD Training')
    
    # Data settings
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--label_ratio', type=float, default=0.05,
                        help='Ratio of labeled samples (default: 0.05)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size (default: 256)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    
    # Model settings
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained ResNet-50')
    
    # UAPA settings
    parser.add_argument('--window_size', type=int, default=16,
                        help='Window size for UAPA (default: 16)')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Protection ratio for UAPA (default: 0.3)')
    parser.add_argument('--rho_max', type=float, default=0.5,
                        help='Maximum paste ratio for UAPA (default: 0.5)')
    parser.add_argument('--rho_min', type=float, default=0.1,
                        help='Minimum paste ratio for UAPA (default: 0.1)')
    
    # DRCL settings
    parser.add_argument('--num_anchors', type=int, default=32,
                        help='Number of anchors for DRCL (default: 32)')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples per anchor for DRCL (default: 64)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for contrastive learning (default: 0.1)')
    parser.add_argument('--memory_size', type=int, default=256,
                        help='Memory bank size for DRCL (default: 256)')
    
    # UGLR settings
    parser.add_argument('--gamma_labeled', type=float, default=2.0,
                        help='Gamma for labeled data (default: 2.0)')
    parser.add_argument('--gamma_unlabeled', type=float, default=-1.0,
                        help='Gamma for unlabeled data (default: -1.0)')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Weight for contrastive loss (default: 0.1)')
    parser.add_argument('--confidence_threshold', type=float, default=0.9,
                        help='Confidence threshold for pseudo labels (default: 0.9)')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--warmup_epochs', type=int, default=30,
                        help='Number of supervised warmup epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--lr_end_factor', type=float, default=0.0001,
                        help='End learning rate factor (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--ema_momentum', type=float, default=0.999,
                        help='EMA momentum for teacher update (default: 0.999)')
    
    # Other settings
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validation interval (default: 5)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer and start training
    trainer = USCDTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
