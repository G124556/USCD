"""
USCD Testing Script
Evaluate trained model on test dataset
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from model.uscd_model import USCDModel
from dataset import create_dataloaders


class USCDTester:
    """
    USCD Model Tester
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        print("Loading USCD model...")
        self.model = USCDModel(
            num_classes=args.num_classes,
            pretrained=False
        ).to(self.device)
        
        # Load checkpoint
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {args.checkpoint}")
            if 'metrics' in checkpoint:
                print(f"Checkpoint metrics: {checkpoint['metrics']}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
        # Create dataloader
        print(f"Loading test data from {args.data_root}...")
        _, _, self.test_loader = create_dataloaders(
            root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_ratio=1.0,  # Use all data for testing
            image_size=args.image_size
        )
        
        # Create output directory for visualizations
        if args.save_vis:
            os.makedirs(args.vis_dir, exist_ok=True)
        
        print(f"Testing on device: {self.device}")
    
    @torch.no_grad()
    def test(self):
        """
        Run inference on test set and compute metrics
        """
        self.model.eval()
        
        tp, fp, fn, tn = 0, 0, 0, 0
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        print("\nRunning inference on test set...")
        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            img_a = batch['img_a'].to(self.device)
            img_b = batch['img_b'].to(self.device)
            labels = batch['label'].to(self.device)
            names = batch['name']
            
            # Student prediction
            pred, features = self.model(img_a, img_b, is_student=True)
            pred_class = torch.argmax(pred, dim=1)
            
            # Compute uncertainty
            uncertainty = self.model.compute_uncertainty(pred)
            
            # Store predictions
            all_predictions.append(pred_class.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())
            
            # Compute metrics
            tp += ((pred_class == 1) & (labels == 1)).sum().item()
            fp += ((pred_class == 1) & (labels == 0)).sum().item()
            fn += ((pred_class == 0) & (labels == 1)).sum().item()
            tn += ((pred_class == 0) & (labels == 0)).sum().item()
            
            # Save visualizations
            if self.args.save_vis and batch_idx < self.args.max_vis:
                self.visualize_results(
                    img_a, img_b, labels, pred_class, uncertainty,
                    names, batch_idx
                )
        
        # Concatenate all results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)
        
        # Compute overall metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        metrics = {
            'True Positive': tp,
            'False Positive': fp,
            'False Negative': fn,
            'True Negative': tn,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'IoU': iou,
            'Accuracy': accuracy
        }
        
        return metrics, all_predictions, all_labels, all_uncertainties
    
    def visualize_results(self, img_a, img_b, labels, pred, uncertainty, names, batch_idx):
        """
        Visualize predictions and save to file
        """
        batch_size = img_a.size(0)
        
        for i in range(batch_size):
            # Denormalize images
            img_a_np = img_a[i].cpu().numpy().transpose(1, 2, 0)
            img_b_np = img_b[i].cpu().numpy().transpose(1, 2, 0)
            
            # Normalize to [0, 1] for visualization
            img_a_np = (img_a_np - img_a_np.min()) / (img_a_np.max() - img_a_np.min() + 1e-8)
            img_b_np = (img_b_np - img_b_np.min()) / (img_b_np.max() - img_b_np.min() + 1e-8)
            
            label_np = labels[i].cpu().numpy()
            pred_np = pred[i].cpu().numpy()
            uncertainty_np = uncertainty[i].cpu().numpy()
            
            # Create figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Plot images
            axes[0, 0].imshow(img_a_np)
            axes[0, 0].set_title('Pre-temporal Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(img_b_np)
            axes[0, 1].set_title('Post-temporal Image')
            axes[0, 1].axis('off')
            
            # Plot ground truth
            axes[0, 2].imshow(label_np, cmap='gray')
            axes[0, 2].set_title('Ground Truth')
            axes[0, 2].axis('off')
            
            # Plot prediction
            axes[1, 0].imshow(pred_np, cmap='gray')
            axes[1, 0].set_title('Prediction')
            axes[1, 0].axis('off')
            
            # Plot uncertainty
            im = axes[1, 1].imshow(uncertainty_np, cmap='viridis')
            axes[1, 1].set_title('Uncertainty Map')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])
            
            # Plot error map (FP in red, FN in green)
            error_map = np.zeros((*pred_np.shape, 3))
            fp_mask = (pred_np == 1) & (label_np == 0)
            fn_mask = (pred_np == 0) & (label_np == 1)
            tp_mask = (pred_np == 1) & (label_np == 1)
            
            error_map[fp_mask] = [1, 0, 0]  # False positive: red
            error_map[fn_mask] = [0, 1, 0]  # False negative: green
            error_map[tp_mask] = [1, 1, 1]  # True positive: white
            
            axes[1, 2].imshow(error_map)
            axes[1, 2].set_title('Error Map (FP: Red, FN: Green)')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(self.args.vis_dir, f'{names[i]}_result.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def print_metrics(self, metrics):
        """
        Print metrics in a formatted table
        """
        print("\n" + "="*60)
        print("Test Results")
        print("="*60)
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(f"{'':15} {'Predicted Neg':>15} {'Predicted Pos':>15}")
        print(f"{'Actual Neg':15} {metrics['True Negative']:>15} {metrics['False Positive']:>15}")
        print(f"{'Actual Pos':15} {metrics['False Negative']:>15} {metrics['True Positive']:>15}")
        
        # Performance metrics
        print("\nPerformance Metrics:")
        print(f"{'Metric':20} {'Value':>15}")
        print("-" * 35)
        print(f"{'Precision':20} {metrics['Precision']:>15.4f}")
        print(f"{'Recall':20} {metrics['Recall']:>15.4f}")
        print(f"{'F1-Score':20} {metrics['F1-Score']:>15.4f}")
        print(f"{'IoU':20} {metrics['IoU']:>15.4f}")
        print(f"{'Accuracy':20} {metrics['Accuracy']:>15.4f}")
        
        print("="*60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='USCD Testing')
    
    # Data settings
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of test dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size (default: 256)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    
    # Testing settings
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Visualization settings
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization results')
    parser.add_argument('--vis_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--max_vis', type=int, default=20,
                        help='Maximum number of batches to visualize')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create tester
    tester = USCDTester(args)
    
    # Run testing
    metrics, predictions, labels, uncertainties = tester.test()
    
    # Print results
    tester.print_metrics(metrics)
    
    if args.save_vis:
        print(f"Visualizations saved to {args.vis_dir}")


if __name__ == "__main__":
    main()
