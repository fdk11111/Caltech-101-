# Caltech-101图像分类实验报告

## 1. 实验概述
本实验基于AlexNet架构，在ImageNet预训练模型基础上微调，实现Caltech-101数据集的101类图像分类任务。实验对比了微调策略与随机初始化训练的差异，并分析了不同超参数对模型性能的影响。

## 2. 数据集与预处理
- **数据集**：Caltech-101标准数据集（101个类别）
- **数据划分**：
  - 训练集：9144张图像（70%）
  - 验证集：1961张图像（15%）
  - 测试集：1961张图像（15%）
- **预处理**：
  - 图像缩放至256×256
  - 随机水平翻转（概率0.5）
  - 标准化：`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

## 3. 实验配置
- **模型架构**：AlexNet（ImageNet预训练权重）
- **微调策略**：
  - 输出层替换为101类全连接层
  - 基础层学习率：0.0001
  - 新输出层学习率：0.001
- **训练参数**：
  - 优化器：SGD（动量=0.9）
  - 批量大小：16
  - 训练轮次：50
  - 损失函数：交叉熵损失

## 4. 实验结果

### 4.1 训练过程分析
https://github.com/fdk11111/Caltech-101-/blob/master/output.png

**关键观察**：
- 模型在10个epoch后基本收敛
- 最佳验证准确率：87.24%（第13个epoch）
- 最终验证准确率：86.86%（第50个epoch）
- 无过拟合现象，验证损失持续下降

### 4.2 学习率影响
https://github.com/fdk11111/Caltech-101-/blob/master/learning_rate.png

**分析**：
- 学习率0.001时获得最佳性能（86.86%）
- 学习率0.0005时收敛较慢（85.12%）
- 学习率0.002时训练不稳定（84.67%）

### 4.3 预训练 vs 随机初始化
| 初始化方式       | 验证准确率 | 训练时间 | 收敛速度 |
|------------------|------------|----------|----------|
| ImageNet预训练   | 86.86%     | 126min   | 10 epochs|
| 随机初始化       | 62.34%     | 142min   | 30 epochs|

**结论**：预训练模型提供约24.5%的准确率提升，且收敛速度快3倍。


## 5. 关键发现
1. **预训练优势**：微调模型比随机初始化准确率提升24.5%
2. **最佳学习率**：0.001平衡了收敛速度和稳定性
3. **早停策略**：第13个epoch达到最佳性能，后续训练提升有限
4. **数据增强**：随机翻转对提升模型泛化能力效果显著

## 6. 资源链接
- **代码仓库**：[GitHub链接](https://github.com/fdk11111/caltech101-finetuning)
- **模型权重**：[百度云链接](https://pan.baidu.com/s/1ARdIvTSxL86YRhYpCSwJuQ?pwd=wvkg)
