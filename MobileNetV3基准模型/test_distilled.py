# # test_distilled.py
# """
# 评估蒸馏后的模型在测试集上的准确率、精确率、召回率和 F1 分数
# 并将结果保存到 output/distilled 目录下
# """
#
# import os
# import sys
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score
#
# # 添加项目根目录，以便导入 config 和模型定义
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# from MobileNetV3基准模型 import config
# from models.mobilenetv3_small import build_small_model
#
# # =========================== 配置 ===========================
# # 蒸馏后模型权重路径
# DISTILLED_WEIGHT = os.path.join(config.ROOT_DIR, "output", "distilled", "student_distilled.pth")
# # 测试集路径（使用 config 中定义的 TEST_DATA_DIR）
# TEST_DIR = config.TEST_DATA_DIR
# # 类别数
# NUM_CLASSES = config.NUM_CLASSES
# # 批大小
# BATCH_SIZE = config.BATCH_SIZE
# # 设备
# DEVICE = torch.device(config.DEVICE)
# # 结果保存路径
# OUTPUT_DIR = os.path.join(config.ROOT_DIR, "output", "distilled")
# RESULT_FILE = os.path.join(OUTPUT_DIR, "test_metrics.txt")
#
#
# def main():
#     print(f"Using device: {DEVICE}")
#     print(f"Test data directory: {TEST_DIR}")
#
#     # =========================== 加载测试集 ===========================
#     # 使用与验证集相同的预处理（无随机增强）
#     test_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#
#     print(f"Test samples: {len(test_dataset)}")
#     print(f"Classes: {test_dataset.classes}")
#
#     # =========================== 加载模型 ===========================
#     model = build_small_model(num_classes=NUM_CLASSES, pretrained=False)
#     state_dict = torch.load(DISTILLED_WEIGHT, map_location=DEVICE)
#     model.load_state_dict(state_dict)
#     model.to(DEVICE)
#     model.eval()
#     print("Distilled model loaded successfully.")
#
#     # =========================== 评估 ===========================
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc="Testing"):
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#
#             all_preds.extend(predicted.cpu().tolist())
#             all_labels.extend(labels.cpu().tolist())
#
#     # 计算各项指标
#     accuracy = accuracy_score(all_labels, all_preds) * 100
#     precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
#         all_labels, all_preds, average='macro', zero_division=0
#     )
#     precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
#         all_labels, all_preds, average='weighted', zero_division=0
#     )
#
#     # 生成结果字符串
#     result_lines = [
#         "=" * 50,
#         f"Test accuracy of distilled model: {accuracy:.2f}%",
#         "--- Macro Average ---",
#         f"Precision: {precision_macro:.4f}",
#         f"Recall:    {recall_macro:.4f}",
#         f"F1-score:  {f1_macro:.4f}",
#         "--- Weighted Average ---",
#         f"Precision: {precision_weighted:.4f}",
#         f"Recall:    {recall_weighted:.4f}",
#         f"F1-score:  {f1_weighted:.4f}",
#         "=" * 50
#     ]
#     result_text = "\n".join(result_lines)
#
#     # 打印到控制台
#     print("\n" + result_text)
#
#     # 保存到文件
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     with open(RESULT_FILE, "w", encoding="utf-8") as f:
#         f.write(result_text)
#     print(f"\nResults saved to: {RESULT_FILE}")
#
#
# if __name__ == "__main__":
#     main()
# test_distilled.py
"""
评估蒸馏后的模型在测试集上的准确率、精确率、召回率和 F1 分数
并输出整体指标和每一类别的指标，结果保存到 output/distilled 目录下
"""

# import os
# import sys
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score
#
# # 添加项目根目录，以便导入 config 和模型定义
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# from MobileNetV3基准模型 import config
# from models.mobilenetv3_small import build_small_model
#
# # =========================== 配置 ===========================
# DISTILLED_WEIGHT = os.path.join(config.ROOT_DIR, "output", "distilled", "student_distilled.pth")
# TEST_DIR = config.TEST_DATA_DIR
# NUM_CLASSES = config.NUM_CLASSES
# BATCH_SIZE = config.BATCH_SIZE
# DEVICE = torch.device(config.DEVICE)
# OUTPUT_DIR = os.path.join(config.ROOT_DIR, "output", "distilled")
# RESULT_FILE = os.path.join(OUTPUT_DIR, "test_metrics.txt")
#
#
# def main():
#     print(f"Using device: {DEVICE}")
#     print(f"Test data directory: {TEST_DIR}")
#
#     # =========================== 加载测试集 ===========================
#     test_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#
#     print(f"Test samples: {len(test_dataset)}")
#     class_names = test_dataset.classes
#     print(f"Classes: {class_names}")
#
#     # =========================== 加载模型 ===========================
#     model = build_small_model(num_classes=NUM_CLASSES, pretrained=False)
#     state_dict = torch.load(DISTILLED_WEIGHT, map_location=DEVICE)
#     model.load_state_dict(state_dict)
#     model.to(DEVICE)
#     model.eval()
#     print("Distilled model loaded successfully.")
#
#     # =========================== 评估 ===========================
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc="Testing"):
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             all_preds.extend(predicted.cpu().tolist())
#             all_labels.extend(labels.cpu().tolist())
#
#     # ---------- 整体指标 ----------
#     accuracy = accuracy_score(all_labels, all_preds) * 100
#     precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
#         all_labels, all_preds, average='macro', zero_division=0
#     )
#     precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
#         all_labels, all_preds, average='weighted', zero_division=0
#     )
#
#     # ---------- 每类指标 ----------
#     # 逐类精确率、召回率、F1、支持度（样本数）
#     prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
#         all_labels, all_preds, labels=range(NUM_CLASSES), zero_division=0
#     )
#     # 逐类准确率 = 该类正确预测数 / 该类样本数 (即召回率，因为每类的准确率就是该类的召回率)
#     acc_per_class = rec_per_class * 100
#
#     # ---------- 生成结果文本 ----------
#     result_lines = []
#     result_lines.append("=" * 70)
#     result_lines.append(f"Overall test accuracy: {accuracy:.2f}%")
#     result_lines.append("")
#     result_lines.append("--- Macro Average ---")
#     result_lines.append(f"Precision: {precision_macro:.4f}")
#     result_lines.append(f"Recall:    {recall_macro:.4f}")
#     result_lines.append(f"F1-score:  {f1_macro:.4f}")
#     result_lines.append("")
#     result_lines.append("--- Weighted Average ---")
#     result_lines.append(f"Precision: {precision_weighted:.4f}")
#     result_lines.append(f"Recall:    {recall_weighted:.4f}")
#     result_lines.append(f"F1-score:  {f1_weighted:.4f}")
#     result_lines.append("")
#     result_lines.append("--- Per-Class Metrics ---")
#     header = f"{'Class':<15} {'Accuracy%':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}"
#     result_lines.append(header)
#     result_lines.append("-" * 70)
#     for idx, name in enumerate(class_names):
#         result_lines.append(
#             f"{name:<15} {acc_per_class[idx]:10.2f} {prec_per_class[idx]:10.4f} "
#             f"{rec_per_class[idx]:10.4f} {f1_per_class[idx]:10.4f} {int(support_per_class[idx]):10d}"
#         )
#     result_lines.append("=" * 70)
#
#     result_text = "\n".join(result_lines)
#
#     # 打印到控制台
#     print("\n" + result_text)
#
#     # 保存到文件
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     with open(RESULT_FILE, "w", encoding="utf-8") as f:
#         f.write(result_text)
#     print(f"\nResults saved to: {RESULT_FILE}")
#
#
# if __name__ == "__main__":
#     main()
# evaluate_distilled_model.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import config

# ------------------------- 配置 -------------------------
CHECKPOINT_PATH = r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型\output\distilled\student_distilled.pth"


# 如果蒸馏模型不是 MobileNetV3-Small，请修改下面的 build_model 函数

def build_model(num_classes):
    """构建模型结构（与训练时完全一致）"""
    # 默认使用 MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ------------------------- 测试集加载 -------------------------
def get_test_loader():
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(root=config.TEST_DATA_DIR, transform=test_transform)
    # 验证类别是否与配置一致
    assert test_dataset.classes == config.CLASS_NAMES, "测试集类别与配置不一致"
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             num_workers=0, pin_memory=True)
    return test_loader, test_dataset.classes


# ------------------------- 评估函数 -------------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    return accuracy, precision, recall, f1


# ------------------------- 计算参数量 -------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------- 主函数 -------------------------
def main():
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}\n")

    # 加载测试集
    test_loader, class_names = get_test_loader()
    num_classes = len(class_names)
    print(f"Test set size: {len(test_loader.dataset)} images, {num_classes} classes\n")

    # 构建模型
    model = build_model(num_classes)
    print(f"Model built: {model.__class__.__name__}")

    # 加载权重
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    # 兼容两种保存格式：直接 state_dict 或 包含 model_state_dict 的字典
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # 可选：打印保存的类别信息（如果有）
        if 'class_names' in checkpoint:
            saved_classes = checkpoint['class_names']
            print(f"Checkpoint saved with classes: {saved_classes}")
            assert saved_classes == class_names, "类别名称不一致，请检查！"
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)

    # 计算参数量
    param_count = count_parameters(model)
    print(f"Total trainable parameters: {param_count:,} ({param_count / 1e6:.2f}M)\n")

    # 评估
    acc, prec, rec, f1 = evaluate_model(model, test_loader, device)

    # 打印结果
    print("=" * 50)
    print("Evaluation Results on Test Set:")
    print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 50)

    # 可选：保存结果到文件
    output_file = os.path.join(os.path.dirname(CHECKPOINT_PATH), "distilled_model_metrics.txt")
    with open(output_file, 'w') as f:
        f.write(f"Model: {CHECKPOINT_PATH}\n")
        f.write(f"Parameters: {param_count} ({param_count / 1e6:.2f}M)\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Precision: {prec:.6f}\n")
        f.write(f"Recall: {rec:.6f}\n")
        f.write(f"F1 Score: {f1:.6f}\n")
    print(f"\nResults also saved to: {output_file}")


if __name__ == '__main__':
    main()