import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import config

# ---------- 模型专用输出路径 ----------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "mobilenetv3_small")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "best_model.pth")
HISTORY_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "history.json")
LOSS_CURVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "loss_curve.png")
# -----------------------------------

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE[0]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def create_data_loaders(train_transform, val_transform):
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DATA_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=config.VAL_DATA_DIR, transform=val_transform)
    assert train_dataset.classes == config.CLASS_NAMES, "类别不一致"
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    return train_loader, val_loader

def build_model(num_classes, device):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(device)
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1} [Train] Batch {batch_idx}/{len(train_loader)} '
                  f'Loss: {loss.item():.4f} Acc: {100 * correct / total:.2f}%')
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if batch_idx % 10 == 0:
                print(f'[Val] Batch {batch_idx}/{len(val_loader)} '
                      f'Loss: {loss.item():.4f} Acc: {100 * correct / total:.2f}%')
    val_loss = running_loss / total
    val_acc = 100 * correct / total
    return val_loss, val_acc

def plot_loss_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (MobileNetV3-Small)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Loss curves saved to {save_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    print(f'Using device: {device}')

    train_transform, val_transform = get_transforms()
    train_loader, val_loader = create_data_loaders(train_transform, val_transform)

    model = build_model(config.NUM_CLASSES, device)
    print(f'Model: MobileNetV3-Small, num_classes={config.NUM_CLASSES}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    start_epoch = 0
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f'Resumed from {args.resume}, epoch {start_epoch}')

    for epoch in range(start_epoch, args.epochs):
        print(f'\n--- Epoch {epoch+1}/{args.epochs} ---')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': config.CLASS_NAMES
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
            print(f'Best model saved with val_acc: {val_acc:.2f}%')

    print(f'\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%')

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'train_accs': train_accs
    }
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=4)
    print(f'Training history saved to {HISTORY_SAVE_PATH}')

    plot_loss_curves(train_losses, val_losses, LOSS_CURVE_PATH)

if __name__ == '__main__':
    main()
