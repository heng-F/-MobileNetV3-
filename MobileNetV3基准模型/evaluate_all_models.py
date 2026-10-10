# evaluate_all_models.py
import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import config

# ------------------------- 模型输出目录配置 -------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_INFO = {
    'MobileNetV3-Small': {
        'checkpoint': os.path.join(ROOT_DIR, 'output', 'mobilenetv3_small', 'best_model.pth'),
        'build_func': lambda num_classes: build_mobilenetv3(num_classes)
    },
    'EfficientNet-B0': {
        'checkpoint': os.path.join(ROOT_DIR, 'output', 'efficientnetb0', 'best_model.pth'),
        'build_func': lambda num_classes: build_efficientnetb0(num_classes)
    },
    'ShuffleNet V2': {
        'checkpoint': os.path.join(ROOT_DIR, 'output', 'shufflenetv2', 'best_model.pth'),
        'build_func': lambda num_classes: build_shufflenetv2(num_classes)
    },
    'SqueezeNet': {
        'checkpoint': os.path.join(ROOT_DIR, 'output', 'squeezenet', 'best_model.pth'),
        'build_func': lambda num_classes: build_squeezenet(num_classes)
    }
}


# ------------------------- 模型构建函数（与训练脚本一致） -------------------------
def build_mobilenetv3(num_classes):
    model = models.mobilenet_v3_small(weights=None)  # 不加载预训练，后面加载权重
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnetb0(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_shufflenetv2(num_classes):
    model = models.shufflenet_v2_x1_0(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_squeezenet(num_classes):
    model = models.squeezenet1_0(weights=None)
    final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
    model.classifier[1] = final_conv
    model.num_classes = num_classes
    return model


# ------------------------- 测试集数据加载 -------------------------
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
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


# ------------------------- 计算参数量 -------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------- 主函数 -------------------------
def main():
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}\n")

    test_loader, class_names = get_test_loader()
    num_classes = len(class_names)
    print(f"Test set size: {len(test_loader.dataset)} images, {num_classes} classes\n")

    results = []

    for model_name, info in MODEL_INFO.items():
        checkpoint_path = info['checkpoint']
        if not os.path.exists(checkpoint_path):
            print(f"Warning: {checkpoint_path} not found, skipping {model_name}")
            continue

        print(f"Evaluating {model_name}...")
        # 构建模型
        model = info['build_func'](num_classes)
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)  # 兼容只保存 state_dict 的情况
        model = model.to(device)

        # 计算参数量
        param_count = count_parameters(model)
        # 评估指标
        metrics = evaluate_model(model, test_loader, device)

        results.append({
            'Model': model_name,
            'Parameters (M)': param_count / 1e6,
            **metrics
        })
        print(f"  {model_name} - Params: {param_count / 1e6:.2f}M, Accuracy: {metrics['Accuracy']:.4f}, "
              f"Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}\n")

    # 保存结果到 CSV
    df = pd.DataFrame(results)
    output_path = os.path.join(ROOT_DIR, 'model_comparison_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # 打印表格
    print("\n" + "=" * 80)
    print("Summary Table:")
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == '__main__':
    main()