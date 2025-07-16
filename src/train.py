#-*- codeing = utf-8 -*-
#中山大学国际金融学院
#经管四班冯定康
#@Time: 2025/5/29  16:18
#@File: train.py
import torch
import argparse
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from model import AlexNet101
from data_loader import get_dataloaders


def main():
    # 参数解析
    print("\n 训练脚本已启动！正在初始化...\n")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/caltech-101')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()

    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet101(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # 分层学习率优化器（作业要求3）
    optimizer = optim.SGD([
        {'params': model.alexnet.classifier[6].parameters(), 'lr': args.lr},
        {'params': model.alexnet.features.parameters(), 'lr': args.lr * 0.1}
    ], momentum=0.9)

    # 数据加载
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    # TensorBoard记录
    writer = SummaryWriter(args.log_dir)

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证模式
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100 * correct / total

        # 记录指标
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), './weights/best_model.pth')
            best_val_acc = val_acc

        print(f'Epoch {epoch + 1}/{args.epochs} | '
              f'Train Loss: {train_loss / len(train_loader):.4f} | '
              f'Val Loss: {val_loss / len(val_loader):.4f} | '
              f'Val Acc: {val_acc:.2f}%')


if __name__ == '__main__':
    main()