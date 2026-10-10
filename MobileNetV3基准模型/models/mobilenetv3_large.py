import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights  # 改为Large版权重类
from MobileNetV3基准模型.config import NUM_CLASSES

def get_mobilenetv3_model(pretrained=True):
    """
    加载MobileNetV3-Large模型，适配torchvision新权重参数
    :param pretrained: 是否加载ImageNet预训练权重
    :return: 适配12类垃圾的模型 + 预处理规则（可选）
    """
    # 新写法：用weights参数替代pretrained（Large版）
    if pretrained:
        # 使用官方预训练权重（ImageNet1K）
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)  # 改为large
        # 自动加载权重对应的预处理规则（增强鲁棒性）
        preprocess = weights.transforms()
    else:
        # 不加载预训练权重
        model = models.mobilenet_v3_large(weights=None)  # 改为large
        preprocess = None

    # 修改最后一层全连接层（Large和Small的classifier结构一致，此处无需改）
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

    return model, preprocess  # 新增返回preprocess，方便后续推理

if __name__ == "__main__":
    # 测试模型加载
    model, preprocess = get_mobilenetv3_model(pretrained=True)
    print("MobileNetV3-Large 模型结构：\n", model)
    # 测试输入维度（batch=1, 3通道, 224x224）
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"模型输出维度：{output.shape}（应为 [1, {NUM_CLASSES}]）")
    if preprocess:
        print("预处理规则：\n", preprocess)