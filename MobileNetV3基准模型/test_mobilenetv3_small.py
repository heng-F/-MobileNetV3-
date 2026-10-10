# test_distilled.py
"""
评估蒸馏后的模型在测试集上的准确率、精确率、召回率和 F1 分数
并输出整体指标和每一类别的指标，结果保存到 output/distilled 目录下
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 添加项目根目录，以便导入 config 和模型定义
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MobileNetV3基准模型 import config
from models.mobilenetv3_small import build_small_model

# =========================== 配置 ===========================
DISTILLED_WEIGHT = os.path.join(config.ROOT_DIR, "output", "mobilenetv3_small", "best_model.pth")
TEST_DIR = config.TEST_DATA_DIR
NUM_CLASSES = config.NUM_CLASSES
BATCH_SIZE = config.BATCH_SIZE
DEVICE = torch.device(config.DEVICE)
OUTPUT_DIR = os.path.join(config.ROOT_DIR, "output", "mobilenetv3_small")
RESULT_FILE = os.path.join(OUTPUT_DIR, "test_metrics.txt")


def main():
    print(f"Using device: {DEVICE}")
    print(f"Test data directory: {TEST_DIR}")

    # =========================== 加载测试集 ===========================
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Test samples: {len(test_dataset)}")
    class_names = test_dataset.classes
    print(f"Classes: {class_names}")

    # =========================== 加载模型 ===========================
    model = build_small_model(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(DISTILLED_WEIGHT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])  # 提取模型权重部分
    model.to(DEVICE)
    model.eval()
    print("Distilled model loaded successfully.")

    # =========================== 评估 ===========================
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # ---------- 整体指标 ----------
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    # ---------- 每类指标 ----------
    # 逐类精确率、召回率、F1、支持度（样本数）
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(NUM_CLASSES), zero_division=0
    )
    # 逐类准确率 = 该类正确预测数 / 该类样本数 (即召回率，因为每类的准确率就是该类的召回率)
    acc_per_class = rec_per_class * 100

    # ---------- 生成结果文本 ----------
    result_lines = []
    result_lines.append("=" * 70)
    result_lines.append(f"Overall test accuracy: {accuracy:.2f}%")
    result_lines.append("")
    result_lines.append("--- Macro Average ---")
    result_lines.append(f"Precision: {precision_macro:.4f}")
    result_lines.append(f"Recall:    {recall_macro:.4f}")
    result_lines.append(f"F1-score:  {f1_macro:.4f}")
    result_lines.append("")
    result_lines.append("--- Weighted Average ---")
    result_lines.append(f"Precision: {precision_weighted:.4f}")
    result_lines.append(f"Recall:    {recall_weighted:.4f}")
    result_lines.append(f"F1-score:  {f1_weighted:.4f}")
    result_lines.append("")
    result_lines.append("--- Per-Class Metrics ---")
    header = f"{'Class':<15} {'Accuracy%':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}"
    result_lines.append(header)
    result_lines.append("-" * 70)
    for idx, name in enumerate(class_names):
        result_lines.append(
            f"{name:<15} {acc_per_class[idx]:10.2f} {prec_per_class[idx]:10.4f} "
            f"{rec_per_class[idx]:10.4f} {f1_per_class[idx]:10.4f} {int(support_per_class[idx]):10d}"
        )
    result_lines.append("=" * 70)

    result_text = "\n".join(result_lines)

    # 打印到控制台
    print("\n" + result_text)

    # 保存到文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"\nResults saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
