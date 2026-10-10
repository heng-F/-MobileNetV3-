# models/mobilenetv3_small.py
import torch.nn as nn
from torchvision import models

def build_small_model(num_classes, pretrained=True):
    """
    构建 MobileNetV3-Small 模型并替换分类头
    :param num_classes: 分类类别数
    :param pretrained: 是否加载 ImageNet 预训练权重
    :return: MobileNetV3-Small 模型实例
    """
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)

    # 替换分类器最后一层
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model