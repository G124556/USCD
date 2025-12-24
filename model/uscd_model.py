"""
USCD: Uncertainty-Guided Semi-Supervised Change Detection
Main Network Architecture Implementation

Paper: "Uncertainty-Guided Semi-Supervised Change Detection for Remote Sensing Images"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DeepLabDecoder(nn.Module):
    """
    DeepLab decoder with ASPP (Atrous Spatial Pyramid Pooling)
    """
    def __init__(self, in_channels=2048, num_classes=2, rates=[6, 12, 18]):
        super(DeepLabDecoder, self).__init__()
        
        # ASPP module
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Concatenation and final layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(256 * 5, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.conv2 = nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class ChangeDetectionNetwork(nn.Module):
    """
    Change Detection Network with ResNet-50 encoder and DeepLab decoder
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(ChangeDetectionNetwork, self).__init__()
        
        # ResNet-50 encoder
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract layers (modify first conv to accept 6 channels for bi-temporal images)
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Initialize with pretrained weights (duplicate for 6 channels)
            pretrained_weight = resnet.conv1.weight.data
            self.conv1.weight.data[:, :3, :, :] = pretrained_weight
            self.conv1.weight.data[:, 3:, :, :] = pretrained_weight
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # DeepLab decoder
        self.decoder = DeepLabDecoder(in_channels=2048, num_classes=num_classes)
        
    def forward(self, x1, x2):
        """
        Args:
            x1: Pre-temporal image [B, 3, H, W]
            x2: Post-temporal image [B, 3, H, W]
        Returns:
            output: Change prediction [B, 2, H, W]
            features: Extracted features for contrastive learning
        """
        # Concatenate bi-temporal images
        x = torch.cat([x1, x2], dim=1)  # [B, 6, H, W]
        
        input_size = x.size()[2:]
        
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)  # [B, 2048, H/32, W/32]
        
        # Decoder
        output = self.decoder(features)  # [B, 2, H/32, W/32]
        
        # Upsample to input size
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        return output, features
    
    def extract_features(self, x1, x2):
        """
        Extract features without final classification layer
        """
        x = torch.cat([x1, x2], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)
        
        return features


class USCDModel(nn.Module):
    """
    Complete USCD Model with teacher-student architecture
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(USCDModel, self).__init__()
        
        # Student network
        self.student = ChangeDetectionNetwork(num_classes=num_classes, pretrained=pretrained)
        
        # Teacher network (same architecture)
        self.teacher = ChangeDetectionNetwork(num_classes=num_classes, pretrained=pretrained)
        
        # Initialize teacher with student weights
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False  # Teacher parameters are not optimized
        
        self.num_classes = num_classes
        
    def forward(self, x1, x2, is_student=True):
        """
        Forward pass through student or teacher network
        """
        if is_student:
            return self.student(x1, x2)
        else:
            with torch.no_grad():
                return self.teacher(x1, x2)
    
    @torch.no_grad()
    def update_teacher(self, momentum=0.999):
        """
        Update teacher network using EMA: θ_t = α*θ_t + (1-α)*θ_s
        """
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data
    
    def compute_uncertainty(self, predictions):
        """
        Compute uncertainty map: U(x,y) = 1 - |P0(x,y) - P1(x,y)|
        
        Args:
            predictions: [B, 2, H, W] logits
        Returns:
            uncertainty: [B, H, W] uncertainty map
        """
        probs = F.softmax(predictions, dim=1)
        p0 = probs[:, 0, :, :]  # Unchanged class
        p1 = probs[:, 1, :, :]  # Changed class
        
        uncertainty = 1.0 - torch.abs(p0 - p1)
        
        return uncertainty


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = USCDModel(num_classes=2, pretrained=False).to(device)
    
    # Test input
    x1 = torch.randn(2, 3, 256, 256).to(device)
    x2 = torch.randn(2, 3, 256, 256).to(device)
    
    # Forward pass
    output_s, features_s = model(x1, x2, is_student=True)
    output_t, features_t = model(x1, x2, is_student=False)
    
    # Compute uncertainty
    uncertainty = model.compute_uncertainty(output_t)
    
    print(f"Student output shape: {output_s.shape}")
    print(f"Teacher output shape: {output_t.shape}")
    print(f"Features shape: {features_s.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    
    # Update teacher
    model.update_teacher(momentum=0.999)
    
    print("Model test passed!")
