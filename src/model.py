#-*- codeing = utf-8 -*-
#中山大学国际金融学院
#经管四班冯定康
#@Time: 2025/5/29  16:59
#@File: model.py.py
import torch.nn as nn
from torchvision import models


class AlexNet101(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        # 加载预训练AlexNet
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)

        # 修改输出层（关键修改）
        self.alexnet.classifier[6] = nn.Linear(4096, 101)  # 101类输出

        # 冻结非输出层参数
        if freeze_backbone:
            for param in self.alexnet.parameters():
                param.requires_grad = False
            # 解冻分类器最后一层
            for param in self.alexnet.classifier[6].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.alexnet(x)