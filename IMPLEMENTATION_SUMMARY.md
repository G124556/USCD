# USCD 完整实现总结

## 📋 项目概述

这是对论文《Uncertainty-Guided Semi-Supervised Change Detection for Remote Sensing Images》的完整PyTorch实现。该实现严格遵循论文中描述的所有细节和算法。

## 📂 文件结构

```
USCD/
├── uscd_model.py          # 主网络模型 (ResNet-50 + DeepLab + Teacher-Student)
├── uapa_module.py         # UAPA模块 (不确定性感知的保护性数据增强)
├── drcl_module.py         # DRCL模块 (困难区域对比学习)
├── uglr_module.py         # UGLR模块 (不确定性引导的损失重加权)
├── dataset.py             # 数据集加载器
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── quick_test.py          # 快速验证脚本
├── requirements.txt       # 依赖包列表
└── README.md              # 详细使用说明
```

## 🎯 核心实现细节

### 1. 主网络架构 (uscd_model.py)

**网络结构:**
- **编码器**: ResNet-50 (修改第一层卷积接受6通道输入以处理双时相图像)
- **解码器**: DeepLab with ASPP (Atrous Spatial Pyramid Pooling)
- **输出**: 2类分类 (unchanged/changed)

**Teacher-Student框架:**
```python
# Teacher参数更新 (论文中的EMA策略)
θ_t = 0.999 × θ_t + 0.001 × θ_s
```

**不确定性度量:**
```python
U(x,y) = 1 - |P₀(x,y) - P₁(x,y)|
```

### 2. UAPA模块 (uapa_module.py)

**四个核心步骤:**

1. **窗口级难度计算** (第一步):
   ```python
   Si,j = (1/|Wi,j|) × Σ U(x,y)  # 计算每个窗口的平均不确定性
   ```

2. **保护区域选择** (第二步):
   ```python
   K = ⌊β × (e/E) × N²⌋  # 动态保护区域数量
   # β=0.3, 随训练进度增加保护区域
   ```

3. **Copy-Paste源区域选择** (第三步):
   ```python
   Di,j = (1/|Wi,j|) × Σ P₁(x,y)  # 选择高变化密度区域
   Nw = ⌊ρ × |A|⌋  # ρ随训练逐渐减小
   ρ = ρmax × (1 - e/E) + ρmin
   ```

4. **自适应混合增强** (第四步):
   ```python
   Xmix = M ⊙ Xa + (1-M) ⊙ Xb  # M是二值掩码
   # 保护区域: M=1 (保持原样)
   # Paste区域: M=0 (复制粘贴)
   ```

### 3. DRCL模块 (drcl_module.py)

**双层对比学习:**

1. **可靠区域识别:**
   ```python
   R(x,y) = I[P_ori(x,y) = P_aug(x,y)]  # 预测一致性
   ```

2. **Local Contrastive Learning:**
   - 随机选择Nr=32个reliable pixels作为anchors
   - 对每个anchor，选择top-Ns=64个高不确定性样本
   - 使用InfoNCE损失

3. **Global Contrastive Learning:**
   - 维护FIFO memory banks (大小256)
   - 存储前景和背景原型向量
   - 当前batch原型与历史原型对比

**InfoNCE损失:**
```python
L = -log[Σ exp(q·p⁺/τ) / (Σ exp(q·p⁺/τ) + Σ exp(q·p⁻/τ))]
# τ=0.1
```

### 4. UGLR模块 (uglr_module.py)

**差异化权重策略:**

1. **标注数据** (增强高不确定性区域学习):
   ```python
   wL(x,y) = exp(γL × U(x,y))  # γL = 2.0 > 0
   ```

2. **未标注数据** (抑制高不确定性区域噪声):
   ```python
   wU(x,y) = exp(γU × U(x,y))  # γU = -1.0 < 0
   ```

3. **加权交叉熵损失:**
   ```python
   L = Σ [w(x,y) / Σw(x',y')] × CE(P(x,y), Y(x,y))
   ```

### 5. 总损失函数

```python
Ltotal = Lsup + Lunsup + 0.1 × Lcontrast

其中:
- Lsup: 监督损失 (高不确定性区域权重大)
- Lunsup: 无监督损失 (低不确定性区域权重大)
- Lcontrast: 对比学习损失 (Llocal + 0.5 × Lglobal)
```

## 🔧 训练流程 (严格遵循Algorithm 1)

### Phase 1: 监督预热 (Epoch 1-30)

```python
for epoch in range(1, 31):
    # 只使用标注数据
    student_pred, features = model.student(labeled_data)
    teacher_pred = model.teacher(labeled_data)
    uncertainty = compute_uncertainty(teacher_pred)
    
    # 只计算监督损失
    loss = UGLR.supervised_loss(student_pred, labels, uncertainty)
    
    # 更新
    loss.backward()
    optimizer.step()
    model.update_teacher(momentum=0.999)
```

### Phase 2: 半监督学习 (Epoch 31-100)

```python
for epoch in range(31, 101):
    # 1. 标注数据处理
    student_pred_l, features_l = model.student(labeled_data)
    teacher_pred_l = model.teacher(labeled_data)
    uncertainty_l = compute_uncertainty(teacher_pred_l)
    
    # 2. 未标注数据处理
    teacher_pred_u = model.teacher(unlabeled_data)
    pseudo_labels = argmax(teacher_pred_u)
    uncertainty_u = compute_uncertainty(teacher_pred_u)
    
    # 3. UAPA增强
    mixed_data, mixed_labels = UAPA(
        unlabeled_data, pseudo_labels, 
        teacher_pred_u, uncertainty_u
    )
    
    # 4. 学生预测增强数据
    student_pred_u, features_u = model.student(mixed_data)
    
    # 5. DRCL对比学习
    reliable_mask = identify_reliable(teacher_pred)
    contrastive_loss = DRCL(
        features_l, teacher_pred_l, 
        uncertainty_l, reliable_mask, labels
    )
    
    # 6. UGLR总损失
    total_loss = UGLR(
        student_pred_l, labels, uncertainty_l,
        student_pred_u, mixed_labels, uncertainty_u,
        contrastive_loss
    )
    
    # 7. 更新
    total_loss.backward()
    clip_grad_norm(model.student.parameters(), max_norm=1.0)
    optimizer.step()
    model.update_teacher(momentum=0.999)
```

## 📊 超参数设置 (论文Table)

| 参数 | 值 | 说明 |
|------|-----|------|
| **训练设置** |
| Total epochs | 100 | 总训练轮数 |
| Warmup epochs | 30 | 监督预热轮数 |
| Batch size | 8 | 批次大小 |
| Initial LR | 0.01 | 初始学习率 |
| LR decay | linear | 线性衰减到1e-4 |
| Momentum | 0.9 | SGD动量 |
| Weight decay | 1e-4 | 权重衰减 |
| **UAPA参数** |
| Window size | 16 | 窗口大小 |
| β | 0.3 | 保护比率 |
| ρ_max | 0.5 | 最大粘贴比率 |
| ρ_min | 0.1 | 最小粘贴比率 |
| **DRCL参数** |
| N_r | 32 | Anchor数量 |
| N_s | 64 | 每个anchor的样本数 |
| τ | 0.1 | 温度参数 |
| Memory size | 256 | Memory bank大小 |
| **UGLR参数** |
| γ_L | 2.0 | 标注数据权重系数 |
| γ_U | -1.0 | 未标注数据权重系数 |
| Confidence | 0.9 | 伪标签置信度阈值 |
| **EMA** |
| α | 0.999 | Teacher更新动量 |

## 🎯 实现特点

### ✅ 完全遵循论文

1. **网络架构**: ResNet-50 + DeepLab (Section III.A)
2. **不确定性度量**: U = 1 - |P₀ - P₁| (Equation 2)
3. **UAPA**: 四步流程 (Section III.B, Equations 3-12)
4. **DRCL**: 双层对比学习 (Section III.C, Equations 13-17)
5. **UGLR**: 差异化权重 (Section III.D, Equations 18-22)
6. **训练流程**: Algorithm 1完整实现

### ✅ 代码质量

1. **模块化设计**: 每个组件独立实现
2. **详细注释**: 标注论文对应的公式和章节
3. **类型提示**: 清晰的参数说明
4. **错误处理**: 边界情况处理
5. **测试代码**: quick_test.py验证所有模块

### ✅ 易用性

1. **命令行接口**: 完整的argparse参数
2. **日志记录**: TensorBoard支持
3. **检查点保存**: 最佳模型和最新模型
4. **可视化**: 测试时保存预测结果
5. **详细README**: 完整的使用说明

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

```
dataset/
├── train/
│   ├── A/      # 前时相图像
│   ├── B/      # 后时相图像
│   └── label/  # 变化标签
├── val/
└── test/
```

### 3. 训练模型

```bash
# 5%标注数据
python train.py \
    --data_root ./dataset \
    --label_ratio 0.05 \
    --epochs 100 \
    --batch_size 8 \
    --pretrained
```

### 4. 测试模型

```bash
python test.py \
    --data_root ./dataset \
    --checkpoint ./checkpoints/best.pth \
    --save_vis
```

### 5. 快速验证

```bash
# 验证实现是否正确(不需要数据集)
python quick_test.py
```

## 📈 预期性能

根据论文Table II，在不同数据集上的预期性能 (5%标注数据):

| 数据集 | F1-Score | IoU | 特点 |
|--------|----------|-----|------|
| LEVIR-CD | 90.22% | 81.56% | 建筑物变化 |
| WHU-CD | 89.41% | 81.52% | 建筑物变化 |
| CDD | 87.27% | 79.02% | 季节性变化 |
| S2Looking | 55.82% | 39.18% | 多尺度卫星 |
| SYSU-CD | 79.8% | 66.5% | 城市区域 |
| JL1-CD | 71.15% | 55.23% | 高分辨率卫星 |

## 🔍 代码验证

所有模块都经过测试:
- ✅ 网络前向传播
- ✅ 不确定性计算
- ✅ Teacher-Student EMA更新
- ✅ UAPA窗口划分和增强
- ✅ DRCL对比学习
- ✅ UGLR损失加权
- ✅ 反向传播和梯度裁剪

运行 `python quick_test.py` 即可验证所有功能。

## 📚 论文对应关系

| 代码文件 | 论文章节 | 关键公式 |
|---------|---------|---------|
| uscd_model.py | Section III.A | Eq. 1-2 |
| uapa_module.py | Section III.B | Eq. 3-12 |
| drcl_module.py | Section III.C | Eq. 13-17 |
| uglr_module.py | Section III.D | Eq. 18-22 |
| train.py | Algorithm 1 | 完整流程 |

## 💡 使用建议

1. **首次使用**: 先运行quick_test.py验证环境
2. **小数据集**: 从5%标注开始，逐步增加
3. **显存不足**: 减小batch_size或image_size
4. **训练时间**: 单个epoch约400秒 (论文Table V)
5. **调优**: 重点调整warmup_epochs和confidence_threshold

## 🎓 学习路径

1. **理解论文**: 阅读论文Section III (Method)
2. **查看架构**: uscd_model.py了解整体结构
3. **学习模块**: 按UAPA → DRCL → UGLR顺序
4. **运行测试**: quick_test.py验证理解
5. **实际训练**: 使用小数据集快速迭代

## 📝 注意事项

1. **数据格式**: 标签必须是二值图像 (0=unchanged, 1=changed)
2. **GPU内存**: 建议至少8GB显存 (batch_size=8, image_size=256)
3. **训练时间**: 100 epochs约11小时 (取决于硬件)
4. **预训练**: 建议使用--pretrained加载ResNet-50预训练权重
5. **随机性**: 使用--seed固定随机种子保证可复现

## 🔧 故障排除

| 问题 | 解决方案 |
|------|---------|
| OOM错误 | 减小batch_size或image_size |
| 训练慢 | 增加num_workers或使用更快GPU |
| 性能差 | 检查数据格式和标签质量 |
| 导入错误 | 安装requirements.txt中的依赖 |

## 📞 支持

如有问题:
1. 查看README.md获取详细说明
2. 运行quick_test.py验证环境
3. 检查论文中的算法描述
4. 查看代码中的详细注释

---

**重要**: 这是对论文的完整实现，所有算法细节都严格遵循原文。代码经过充分测试，可以直接用于研究和实验。

**作者**: 基于IEEE论文"Uncertainty-Guided Semi-Supervised Change Detection for Remote Sensing Images"

**日期**: 2024年12月

**版本**: 1.0
