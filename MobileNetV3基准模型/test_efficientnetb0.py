# test_efficientnetb0.py
"""
评估 EfficientNet-B0 在测试集上的性能，输出整体和每类指标
结果保存到指定绝对路径
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 确保能导入 config（根据你的项目结构调整）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MobileNetV3基准模型 import config

# ==================== 绝对路径（硬编码） ====================
WEIGHT_PATH = r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型\output\efficientnetb0\efficientnetb0_best.pth"
OUTPUT_DIR = r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型\output\efficientnetb0"
RESULT_FILE = os.path.join(OUTPUT_DIR, "test_metrics.txt")

# ==================== 其他配置（来自 config） ====================
TEST_DIR = config.TEST_DATA_DIR          # 测试集根目录
NUM_CLASSES = config.NUM_CLASSES         # 垃圾类别数
BATCH_SIZE = config.BATCH_SIZE           # 评估时的批量大小
DEVICE = torch.device(config.DEVICE)     # 设备

# ==================== 模型构建（与训练时完全一致） ====================
def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)  # 不加载预训练权重，只保留结构
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model

# ==================== 主流程 ====================
def main():
    print(f"Using device: {DEVICE}")
    print(f"Test data directory: {TEST_DIR}")

    # 1. 加载测试集（与验证集同样的预处理）
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    class_names = test_dataset.classes
    print(f"Test samples: {len(test_dataset)}, Classes: {class_names}")

    # 2. 加载模型权重
    model = build_model(NUM_CLASSES)
    checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])  # 检查点中键名为'model_state_dict'
    model.to(DEVICE)
    model.eval()
    print("EfficientNet-B0 model loaded from:", WEIGHT_PATH)

    # 3. 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 4. 计算整体指标
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0)

    # 5. 计算每类指标
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(NUM_CLASSES), zero_division=0)
    acc_per_class = rec_per_class * 100   # 每类准确率 = 召回率

    # 6. 生成结果文本
    lines = []
    lines.append("=" * 70)
    lines.append("Model: EfficientNet-B0")
    lines.append(f"Overall test accuracy: {accuracy:.2f}%")
    lines.append("")
    lines.append("--- Macro Average ---")
    lines.append(f"Precision: {precision_macro:.4f}")
    lines.append(f"Recall:    {recall_macro:.4f}")
    lines.append(f"F1-score:  {f1_macro:.4f}")
    lines.append("")
    lines.append("--- Weighted Average ---")
    lines.append(f"Precision: {precision_weighted:.4f}")
    lines.append(f"Recall:    {recall_weighted:.4f}")
    lines.append(f"F1-score:  {f1_weighted:.4f}")
    lines.append("")
    lines.append("--- Per-Class Metrics ---")
    header = f"{'Class':<15} {'Accuracy%':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}"
    lines.append(header)
    lines.append("-" * 70)
    for idx, name in enumerate(class_names):
        lines.append(
            f"{name:<15} {acc_per_class[idx]:10.2f} {prec_per_class[idx]:10.4f} "
            f"{rec_per_class[idx]:10.4f} {f1_per_class[idx]:10.4f} {int(support_per_class[idx]):10d}"
        )
    lines.append("=" * 70)
    result_text = "\n".join(lines)

    # 7. 保存与打印
    print("\n" + result_text)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"Results saved to: {RESULT_FILE}")

if __name__ == "__main__":
    main()