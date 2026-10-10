# distill.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
from tqdm import tqdm
import argparse

# 导入你的配置和模型定义（需保证路径正确）
from MobileNetV3基准模型.config import TRAIN_DATA_DIR, VAL_DATA_DIR, BATCH_SIZE, DEVICE, NUM_CLASSES
from models.mobilenetv3_large import get_mobilenetv3_model   # Large模型
from models.mobilenetv3_small import build_small_model        # 需要你提供Small模型定义函数，我根据train_small.py给出

# ----------------- 配置参数 -----------------
TEACHER_WEIGHT = r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型\output\baseline\best_model_baseline.pth"
STUDENT_WEIGHT = r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型\output\small\best_small_model.pth"
OUTPUT_DIR = r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型\output\distilled"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DISTILLED_MODEL_PATH = os.path.join(OUTPUT_DIR, "student_distilled.pth")

DISTILL_EPOCHS = 20
TEMPERATURE = 4.0
ALPHA = 0.7      # 蒸馏损失权重，交叉熵权重为 1 - ALPHA
LR = 1e-4
# -------------------------------------------

def get_data_loaders():
    """与train.py中完全一致的数据加载"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DATA_DIR, transform=data_transforms['train']),
        'val': datasets.ImageFolder(VAL_DATA_DIR, transform=data_transforms['val'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    return dataloaders

def build_small_model(num_classes=NUM_CLASSES):
    """根据train_small.py重构Small模型加载函数"""
    from torchvision import models
    model = models.mobilenet_v3_small(weights=None)  # 不加载预训练，后续加载自己的权重
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """计算蒸馏损失"""
    # 硬标签交叉熵
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    # 软标签 KL 散度
    kd_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / T, dim=1),
        nn.functional.softmax(teacher_logits / T, dim=1)
    ) * (alpha * T * T)
    # 总损失
    return (1 - alpha) * ce_loss + kd_loss

def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def main():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # 加载教师模型
    teacher_model, _ = get_mobilenetv3_model(pretrained=False)
    teacher_model.load_state_dict(torch.load(TEACHER_WEIGHT, map_location=device))
    teacher_model.to(device)
    teacher_model.eval()
    print("Teacher model loaded.")

    # 加载学生模型
    student_model = build_small_model()
    checkpoint = torch.load(STUDENT_WEIGHT, map_location=device)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    student_model.to(device)
    print("Student model loaded.")

    # 数据加载器
    dataloaders = get_data_loaders()
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # 优化器
    optimizer = optim.Adam(student_model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DISTILL_EPOCHS)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(student_model.state_dict())

    for epoch in range(DISTILL_EPOCHS):
        student_model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{DISTILL_EPOCHS}')
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            student_logits = student_model(inputs)
            loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        # 验证
        val_acc = validate(student_model, val_loader, device)
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(student_model.state_dict())
            torch.save(best_model_wts, DISTILLED_MODEL_PATH)
            print(f"  -> New best model saved! Acc = {best_acc:.2f}%")

    print(f"Distillation finished. Best val acc: {best_acc:.2f}%")
    print(f"Model saved to {DISTILLED_MODEL_PATH}")

if __name__ == "__main__":
    main()