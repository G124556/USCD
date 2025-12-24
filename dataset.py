"""
Dataset and DataLoader for Change Detection
Supports LEVIR-CD, WHU-CD, CDD, and other datasets
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class ChangeDetectionDataset(Dataset):
    """
    Change Detection Dataset
    
    Expected directory structure:
    root/
        A/  (pre-temporal images)
            img1.png
            img2.png
            ...
        B/  (post-temporal images)
            img1.png
            img2.png
            ...
        label/  (change masks)
            img1.png
            img2.png
            ...
    """
    
    def __init__(self, 
                 root,
                 split='train',
                 transform=None,
                 label_ratio=1.0):
        """
        Args:
            root: Root directory of dataset
            split: 'train', 'val', or 'test'
            transform: Augmentation transforms
            label_ratio: Ratio of labeled samples (for semi-supervised learning)
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.label_ratio = label_ratio
        
        # Set paths
        self.img_a_dir = os.path.join(root, split, 'A')
        self.img_b_dir = os.path.join(root, split, 'B')
        self.label_dir = os.path.join(root, split, 'label')
        
        # Get image list
        self.img_list = sorted(os.listdir(self.img_a_dir))
        
        # Determine which samples are labeled
        num_total = len(self.img_list)
        num_labeled = int(num_total * label_ratio)
        
        # Randomly select labeled samples
        indices = np.random.permutation(num_total)
        self.labeled_indices = set(indices[:num_labeled].tolist())
        
        print(f"{split} set: {num_total} total samples, "
              f"{num_labeled} labeled samples ({label_ratio*100:.1f}%)")
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        
        # Load images
        img_a_path = os.path.join(self.img_a_dir, img_name)
        img_b_path = os.path.join(self.img_b_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)
        
        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')
        label = Image.open(label_path)
        
        # Convert to numpy arrays
        img_a = np.array(img_a)
        img_b = np.array(img_b)
        label = np.array(label)
        
        # Binary mask: 0 for unchanged, 1 for changed
        if len(label.shape) == 3:
            label = label[:, :, 0]
        label = (label > 0).astype(np.int64)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=img_a, image2=img_b, mask=label)
            img_a = transformed['image']
            img_b = transformed['image2']
            label = transformed['mask']
        else:
            # Default: to tensor and normalize
            img_a = torch.from_numpy(img_a).permute(2, 0, 1).float() / 255.0
            img_b = torch.from_numpy(img_b).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(label).long()
        
        # Check if this sample is labeled
        is_labeled = idx in self.labeled_indices
        
        return {
            'img_a': img_a,
            'img_b': img_b,
            'label': label,
            'is_labeled': is_labeled,
            'name': img_name
        }


class WeakAugmentation:
    """
    Weak augmentation for change detection
    Includes random flip, rotation, and crop
    """
    def __init__(self, size=256):
        self.size = size
    
    def __call__(self, image, image2, mask):
        import albumentations as A
        
        transform = A.Compose([
            A.RandomCrop(height=self.size, width=self.size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], additional_targets={'image2': 'image'})
        
        transformed = transform(image=image, image2=image2, mask=mask)
        
        # Convert to tensors
        img_a = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
        img_b = torch.from_numpy(transformed['image2']).permute(2, 0, 1).float()
        label = torch.from_numpy(transformed['mask']).long()
        
        return {'image': img_a, 'image2': img_b, 'mask': label}


def create_dataloaders(root, 
                       batch_size=8,
                       num_workers=4,
                       label_ratio=0.05,
                       image_size=256):
    """
    Create train, validation, and test dataloaders
    
    Args:
        root: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of workers for dataloader
        label_ratio: Ratio of labeled samples in training set
        image_size: Input image size
    Returns:
        train_loader, val_loader, test_loader
    """
    # Augmentation
    train_transform = WeakAugmentation(size=image_size)
    val_transform = None  # Will use default transform
    
    # Create datasets
    train_dataset = ChangeDetectionDataset(
        root=root,
        split='train',
        transform=train_transform,
        label_ratio=label_ratio
    )
    
    val_dataset = ChangeDetectionDataset(
        root=root,
        split='val',
        transform=val_transform,
        label_ratio=1.0  # Use all validation samples as labeled
    )
    
    test_dataset = ChangeDetectionDataset(
        root=root,
        split='test',
        transform=val_transform,
        label_ratio=1.0  # Use all test samples as labeled
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset loader...")
    
    # Create a dummy dataset structure for testing
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(tmpdir, split, 'A'), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, split, 'B'), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, split, 'label'), exist_ok=True)
            
            # Create dummy images
            for i in range(10):
                img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
                label = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255)
                
                img.save(os.path.join(tmpdir, split, 'A', f'img_{i}.png'))
                img.save(os.path.join(tmpdir, split, 'B', f'img_{i}.png'))
                label.save(os.path.join(tmpdir, split, 'label', f'img_{i}.png'))
        
        # Test dataloader
        try:
            train_loader, val_loader, test_loader = create_dataloaders(
                root=tmpdir,
                batch_size=2,
                num_workers=0,
                label_ratio=0.5,
                image_size=256
            )
            
            # Test one batch
            batch = next(iter(train_loader))
            print(f"Batch keys: {batch.keys()}")
            print(f"img_a shape: {batch['img_a'].shape}")
            print(f"img_b shape: {batch['img_b'].shape}")
            print(f"label shape: {batch['label'].shape}")
            print(f"is_labeled: {batch['is_labeled']}")
            
            print("\nDataset test passed!")
            
        except Exception as e:
            print(f"Dataset test failed: {e}")
            print("Note: Install albumentations for augmentation support:")
            print("  pip install albumentations")
