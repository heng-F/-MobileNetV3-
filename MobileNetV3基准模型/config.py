# 全局配置文件
import os
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径
TRAIN_DATA_DIR = os.path.join(ROOT_DIR, "data/train")
VAL_DATA_DIR = os.path.join(ROOT_DIR, "data/val")
TEST_DATA_DIR = os.path.join(ROOT_DIR, "data/test")

# 输出根目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "small")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型保存路径
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_small_model.pth")

# 12类垃圾类别
CLASS_NAMES = [
    "battery", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]
NUM_CLASSES = len(CLASS_NAMES)

# 超参数
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (224, 224)

