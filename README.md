# USCD: Uncertainty-Guided Semi-Supervised Change Detection

This is a complete PyTorch implementation of the paper **"Uncertainty-Guided Semi-Supervised Change Detection for Remote Sensing Images"**.

## Overview

USCD is a novel framework for semi-supervised change detection that introduces a triple-guarantee mechanism for targeted learning reinforcement in difficult regions based on systematic uncertainty quantification.

### Key Components

1. **UAPA (Uncertainty-Aware Protective Augmentation)**: Identifies high-uncertainty regions and implements selective copy-paste augmentation
2. **DRCL (Difficult Region Contrastive Learning)**: Enhances feature discriminability through local and global contrastive learning
3. **UGLR (Uncertainty-Guided Loss Re-weighting)**: Applies exponential adaptive weighting for supervision signals

## Architecture

```
USCD Framework
├── uscd_model.py          # Main network (ResNet-50 + DeepLab)
├── uapa_module.py         # Uncertainty-Aware Protective Augmentation
├── drcl_module.py         # Difficult Region Contrastive Learning
├── uglr_module.py         # Uncertainty-Guided Loss Re-weighting
├── dataset.py             # Dataset loader for change detection
├── train.py               # Training script
└── test.py                # Testing script
```

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.10+
- CUDA 10.2+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd uscd

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The expected dataset structure:

```
dataset_root/
├── train/
│   ├── A/           # Pre-temporal images
│   │   ├── img1.png
│   │   └── ...
│   ├── B/           # Post-temporal images
│   │   ├── img1.png
│   │   └── ...
│   └── label/       # Change masks
│       ├── img1.png
│       └── ...
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

### Supported Datasets

- **LEVIR-CD**: Building change detection (0.5m resolution)
- **WHU-CD**: Building change detection (0.2m resolution)
- **CDD**: Seasonal change detection (0.03-1m resolution)
- **S2Looking**: Satellite building changes (0.5-0.8m resolution)
- **SYSU-CD**: Urban area changes (0.5m resolution)
- **JL1-CD**: Jilin-1 satellite changes (0.5-0.75m resolution)

## Training

### Basic Training

```bash
python train.py \
    --data_root /path/to/dataset \
    --label_ratio 0.05 \
    --epochs 100 \
    --batch_size 8 \
    --pretrained \
    --save_dir ./checkpoints \
    --log_dir ./logs
```

### Training with Different Label Ratios

```bash
# 5% labeled data
python train.py --data_root /path/to/dataset --label_ratio 0.05

# 10% labeled data
python train.py --data_root /path/to/dataset --label_ratio 0.10

# 20% labeled data
python train.py --data_root /path/to/dataset --label_ratio 0.20
```

### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--label_ratio` | Ratio of labeled samples | 0.05 |
| `--epochs` | Total training epochs | 100 |
| `--warmup_epochs` | Supervised warmup epochs | 30 |
| `--batch_size` | Batch size | 8 |
| `--lr` | Initial learning rate | 0.01 |
| `--ema_momentum` | EMA momentum for teacher | 0.999 |

### UAPA Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--window_size` | Window size for dividing feature map | 16 |
| `--beta` | Protection ratio | 0.3 |
| `--rho_max` | Maximum paste ratio | 0.5 |
| `--rho_min` | Minimum paste ratio | 0.1 |

### DRCL Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_anchors` | Number of anchor points | 32 |
| `--num_samples` | Number of samples per anchor | 64 |
| `--temperature` | Temperature for InfoNCE loss | 0.1 |
| `--memory_size` | Size of prototype memory banks | 256 |

### UGLR Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gamma_labeled` | Weight coefficient for labeled data | 2.0 |
| `--gamma_unlabeled` | Weight coefficient for unlabeled data | -1.0 |
| `--confidence_threshold` | Threshold for pseudo labels | 0.9 |
| `--contrastive_weight` | Weight for contrastive loss | 0.1 |

## Testing

### Basic Testing

```bash
python test.py \
    --data_root /path/to/dataset \
    --checkpoint ./checkpoints/best.pth \
    --batch_size 8 \
    --save_vis \
    --vis_dir ./visualizations
```

### Testing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint` | Path to model checkpoint | required |
| `--save_vis` | Save visualization results | False |
| `--vis_dir` | Directory for visualizations | ./visualizations |
| `--max_vis` | Max batches to visualize | 20 |

### Output Metrics

The test script will output:
- **Confusion Matrix**: TP, FP, TN, FN
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × Precision × Recall / (Precision + Recall)
- **IoU**: TP / (TP + FP + FN)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)

## Model Architecture Details

### Network Structure

```
Input: Bi-temporal images (Pre-temporal + Post-temporal)
    ↓
ResNet-50 Encoder (6 channels input)
    ↓
DeepLab Decoder with ASPP
    ↓
Output: Change prediction (2 classes)
```

### Teacher-Student Framework

- **Student Network**: Primary learning network
- **Teacher Network**: Provides stable predictions via EMA
- **EMA Update**: θ_t = 0.999 × θ_t + 0.001 × θ_s

### Uncertainty Quantification

```
U(x,y) = 1 - |P₀(x,y) - P₁(x,y)|
```
where P₀ and P₁ are probabilities for unchanged and changed classes.

## Training Process

### Phase 1: Supervised Warmup (Epochs 1-30)

- Use only labeled data
- Train with standard cross-entropy loss
- Build basic change detection capability

### Phase 2: Semi-Supervised Learning (Epochs 31-100)

1. **Generate pseudo-labels** from teacher network
2. **Apply UAPA** for protective augmentation
3. **Compute DRCL loss** for difficult region features
4. **Apply UGLR** for adaptive loss weighting
5. **Update student** via backpropagation
6. **Update teacher** via EMA

## Loss Function

```
L_total = L_sup + L_unsup + 0.1 × L_contrast

where:
- L_sup: Supervised loss with uncertainty weighting
- L_unsup: Unsupervised loss with pseudo labels
- L_contrast: Contrastive learning loss (local + 0.5 × global)
```

## Expected Performance

### LEVIR-CD Dataset (5% labeled data)

| Metric | Performance |
|--------|-------------|
| F1-Score | ~90.22% |
| IoU | ~81.56% |
| Precision | ~90.5% |
| Recall | ~89.9% |

### WHU-CD Dataset (5% labeled data)

| Metric | Performance |
|--------|-------------|
| F1-Score | ~89.41% |
| IoU | ~81.52% |
| Precision | ~90.2% |
| Recall | ~88.7% |

*Note: Results may vary depending on hardware and random initialization.*

## Troubleshooting

### Out of Memory (OOM) Error

```bash
# Reduce batch size
python train.py --batch_size 4

# Reduce image size
python train.py --image_size 224

# Use gradient checkpointing (modify model if needed)
```

### Slow Training

```bash
# Increase number of workers
python train.py --num_workers 8

# Use mixed precision training (add to train.py if needed)
```

### Low Performance

1. Ensure proper data preprocessing
2. Verify label format (binary: 0 for unchanged, 1 for changed)
3. Try different label ratios
4. Adjust hyperparameters (learning rate, warmup epochs)

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{uscd2024,
  title={Uncertainty-Guided Semi-Supervised Change Detection for Remote Sensing Images},
  author={[Authors]},
  journal={IEEE [Journal Name]},
  year={2024}
}
```

## License

This implementation is provided for research purposes only. Please refer to the original paper for licensing details.

## Acknowledgments

This implementation is based on the paper "Uncertainty-Guided Semi-Supervised Change Detection for Remote Sensing Images" and uses:
- PyTorch for deep learning
- ResNet-50 from torchvision
- DeepLab decoder architecture
- Albumentations for data augmentation

## Contact

For questions and issues, please:
1. Check the paper for theoretical details
2. Review the code comments for implementation details
3. Open an issue on the repository

---

**Note**: This is a research implementation. For production use, additional optimization and validation may be required.
