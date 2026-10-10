import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# 导入项目配置与模型定义
from MobileNetV3基准模型.config import *
from models.mobilenetv3_large import get_mobilenetv3_model
from utils.data_utils import get_data_loaders

# ---------- 评估函数 ----------
def evaluate_model(model, dataloader, device):
    """计算模型在给定数据上的准确率、宏平均精确率/召回率/F1、每类指标及预测结果"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 整体准确率
    acc = accuracy_score(all_labels, all_preds)

    # 宏平均指标
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    # 每类指标
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    return {
        'acc': acc,
        'macro_prec': macro_prec,
        'macro_rec': macro_rec,
        'macro_f1': macro_f1,
        'per_class_prec': per_class_prec,
        'per_class_rec': per_class_rec,
        'per_class_f1': per_class_f1,
        'preds': all_preds,
        'labels': all_labels
    }

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=False):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存至 {save_path}")

def plot_per_class_metrics(per_class_prec, per_class_rec, per_class_f1, class_names, save_path):
    """绘制每类精确率、召回率、F1分数的柱状图"""
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, per_class_prec, width, label='Precision')
    rects2 = ax.bar(x, per_class_rec, width, label='Recall')
    rects3 = ax.bar(x + width, per_class_f1, width, label='F1 Score')

    ax.set_ylabel('Scores')
    ax.set_title('Per-class Precision, Recall and F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()

    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"每类指标柱状图已保存至 {save_path}")

def main():
    # 设置设备
    device = torch.device(DEVICE)
    print(f"使用设备: {device}")

    # 输出目录（用于保存评估结果）
    output_dir = os.path.join(ROOT_DIR, "output", "baseline")
    os.makedirs(output_dir, exist_ok=True)

    # ----- 模型权重绝对路径（直接指定）-----
    model_path = r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型\output\baseline\best_model_baseline.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载测试数据
    # 注意：get_data_loaders 需要训练集路径（用于归一化计算）和验证集路径，这里用测试集替代验证集
    _, test_loader = get_data_loaders(
        TRAIN_DATA_DIR, TEST_DATA_DIR, BATCH_SIZE, IMAGE_SIZE
    )
    print(f"测试集样本数: {len(test_loader.dataset)}")

    # 构建模型并加载权重
    model, _ = get_mobilenetv3_model(pretrained=False)  # 不加载预训练，直接加载本地权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("模型加载完成")

    # 评估
    print("\n开始评估基线模型（CrossEntropyLoss）...")
    results = evaluate_model(model, test_loader, device)

    # 打印总体指标
    print("\n========== 测试集评估结果 ==========")
    print(f"准确率 (Accuracy): {results['acc']:.4f}")
    print(f"宏平均精确率 (Macro Precision): {results['macro_prec']:.4f}")
    print(f"宏平均召回率 (Macro Recall): {results['macro_rec']:.4f}")
    print(f"宏平均F1分数 (Macro F1): {results['macro_f1']:.4f}")
    print("\n各类别详细指标：")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:12s} | Precision: {results['per_class_prec'][i]:.4f} | Recall: {results['per_class_rec'][i]:.4f} | F1: {results['per_class_f1'][i]:.4f}")

    # 保存分类报告到文本文件
    report = classification_report(
        results['labels'], results['preds'],
        target_names=CLASS_NAMES, zero_division=0
    )
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n分类报告已保存至 {report_path}")

    # 绘制混淆矩阵
    cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
    plot_confusion_matrix(results['labels'], results['preds'], CLASS_NAMES, cm_path, normalize=False)
    cm_norm_path = os.path.join(output_dir, "test_confusion_matrix_normalized.png")
    plot_confusion_matrix(results['labels'], results['preds'], CLASS_NAMES, cm_norm_path, normalize=True)

    # 绘制每类指标柱状图
    metrics_path = os.path.join(output_dir, "test_per_class_metrics.png")
    plot_per_class_metrics(
        results['per_class_prec'],
        results['per_class_rec'],
        results['per_class_f1'],
        CLASS_NAMES,
        metrics_path
    )

    print("\n评估完成，所有结果已保存至:", output_dir)

if __name__ == "__main__":
    main()

