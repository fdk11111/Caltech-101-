#-*- codeing = utf-8 -*-
#中山大学国际金融学院
#经管四班冯定康
#@Time: 2025/5/29  16:58
#@File: data_loader.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_dataloaders(data_dir='./data/caltech-101', batch_size=64):
    # 数据增强策略（仅训练集）
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 验证/测试集预处理
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(f'{data_dir}/val', transform=val_transform)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    return train_loader, val_loader