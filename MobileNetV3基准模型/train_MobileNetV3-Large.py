import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from MobileNetV3基准模型.config import (
    TRAIN_DATA_DIR, VAL_DATA_DIR, BATCH_SIZE, EPOCHS, DEVICE
)
from models.mobilenetv3_large import get_mobilenetv3_model

# ================== 输出目录 ==================
BASELINE_OUTPUT_DIR = r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型\output\baseline"
os.makedirs(BASELINE_OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH_BASELINE = os.path.join(BASELINE_OUTPUT_DIR, "best_model_baseline.pth")
LOSS_CURVE_PATH = os.path.join(BASELINE_OUTPUT_DIR, "loss_curve_baseline.png")


# ================== 固定随机种子 ==================
def set_seed(seed=42):
    """固定所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已固定为: {seed}")


# ================== 数据预处理（与 CBAM 完全一致） ==================
def get_data_transforms():
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
    return data_transforms


def load_datasets():
    data_transforms = get_data_transforms()
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DATA_DIR, transform=data_transforms['train']),
        'val': datasets.ImageFolder(VAL_DATA_DIR, transform=data_transforms['val'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"类别: {class_names}")
    print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['val']}")
    return dataloaders, dataset_sizes


# ================== 损失曲线绘制 ==================
def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve (Baseline)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"损失曲线已保存至 {save_path}")


# ================== 训练函数 ==================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=EPOCHS):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=f'{phase} batches', unit='batch')
            for inputs, labels in loop:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH_BASELINE)
                print(f'保存新最佳模型，准确率: {best_acc:.4f}')

        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')

    plot_loss_curve(train_loss_history, val_loss_history, LOSS_CURVE_PATH)

    model.load_state_dict(best_model_wts)
    return model


# ================== 主函数 ==================
def main():
    # 固定随机种子
    set_seed(42)

    print(f"使用设备: {DEVICE}")

    # 加载基线模型
    model, _ = get_mobilenetv3_model(pretrained=True)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # 优化器配置与 CBAM 一致：lr=1e-4，无权重衰减
    finetune_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=finetune_lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    dataloaders, dataset_sizes = load_datasets()

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=EPOCHS)

    print(f"基线模型训练完成，最佳模型已保存至 {MODEL_SAVE_PATH_BASELINE}")


if __name__ == "__main__":
    main()