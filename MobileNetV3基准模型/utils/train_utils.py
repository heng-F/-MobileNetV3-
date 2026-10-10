import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, return_predictions=False, num_classes=None):
    """
    验证函数
    当 return_predictions=True 且 num_classes 提供时，返回每类精确率、召回率、F1分数
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total

    if return_predictions:
        # 宏平均指标
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        # 每类指标
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(num_classes)), zero_division=0
        )
        return (val_loss, val_acc, all_preds, all_labels,
                macro_precision, macro_recall, macro_f1,
                per_class_precision, per_class_recall, per_class_f1)
    else:
        return val_loss, val_acc


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
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"混淆矩阵已保存至 {save_path}")


def plot_per_class_metrics(per_class_precision, per_class_recall, per_class_f1, class_names, save_path):
    """
    绘制每类精确率、召回率、F1分数的柱状图
    :param per_class_precision: list, 每类精确率
    :param per_class_recall: list, 每类召回率
    :param per_class_f1: list, 每类F1分数
    :param class_names: list, 类别名称
    :param save_path: 保存路径
    """
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, per_class_precision, width, label='Precision')
    rects2 = ax.bar(x, per_class_recall, width, label='Recall')
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
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"每类指标柱状图已保存至 {save_path}")