import os
import shutil
import random

# 注意：若未提前安装shutil/random，无需额外安装，Python内置

# ========================== 核心配置（根据你的目录修改） ==========================
# 1. 路径配置（当前脚本与raw_data同目录，无需修改）
RAW_DATA_DIR = "raw_data"  # 原始数据集目录（已正确放置在脚本同级）
TARGET_DATA_DIR = "data"  # 拆分后输出目录（生成在脚本同级）

# 2. 12类垃圾类别（严格对应raw_data下的文件夹名，大小写一致！）
CLASS_NAMES = [
    "battery",  # 电池
    "biological",  # 生物/厨余垃圾
    "brown-glass",  # 棕色玻璃
    "cardboard",  # 纸板
    "clothes",  # 衣物
    "green-glass",  # 绿色玻璃
    "metal",  # 金属
    "paper",  # 纸类
    "plastic",  # 塑料
    "shoes",  # 鞋子
    "trash",  # 其他垃圾
    "white-glass"  # 白色玻璃
]

# 3. 拆分比例（保持8:1:1，毕设常用且合理）
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
# ================================================================================

# 检查比例合法性
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "训练/验证/测试集比例和必须为1"


def create_target_dirs():
    """创建train/val/test的目录结构，自动为每个类别建文件夹"""
    for split in ["train", "val", "test"]:
        for cls_name in CLASS_NAMES:
            target_dir = os.path.join(TARGET_DATA_DIR, split, cls_name)
            os.makedirs(target_dir, exist_ok=True)  # exist_ok=True：已存在则不报错
    print(f"✅ 目标目录结构创建完成，输出路径：{os.path.abspath(TARGET_DATA_DIR)}")


def split_single_class(cls_name):
    """拆分单个类别的图片，返回拆分结果"""
    # 原始类别路径
    cls_raw_path = os.path.join(RAW_DATA_DIR, cls_name)

    # 检查原始路径是否存在
    if not os.path.isdir(cls_raw_path):
        print(f"❌ 警告：类别 {cls_name} 的路径 {cls_raw_path} 不存在，跳过该类别")
        return 0, 0, 0

    # 过滤有效图片（支持常见格式，忽略隐藏文件/非图片）
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
    all_imgs = [
        f for f in os.listdir(cls_raw_path)
        if os.path.splitext(f)[1] in img_extensions and not f.startswith(".")
    ]

    # 处理空文件夹情况
    if len(all_imgs) == 0:
        print(f"❌ 警告：类别 {cls_name} 下无有效图片，跳过该类别")
        return 0, 0, 0

    # 随机打乱（固定随机种子，确保拆分结果可复现）
    random.seed(42)  # 毕设建议固定种子，便于复现
    random.shuffle(all_imgs)
    total_imgs = len(all_imgs)

    # 计算各集数量（避免小数误差，优先分配测试集）
    train_num = int(total_imgs * TRAIN_RATIO)
    val_num = int(total_imgs * VAL_RATIO)
    test_num = total_imgs - train_num - val_num  # 兜底，确保总数一致

    # 切片分配
    train_imgs = all_imgs[:train_num]
    val_imgs = all_imgs[train_num:train_num + val_num]
    test_imgs = all_imgs[train_num + val_num:]

    # 复制图片到目标目录
    def copy_imgs(imgs, split_type):
        dst_root = os.path.join(TARGET_DATA_DIR, split_type, cls_name)
        for img in imgs:
            src = os.path.join(cls_raw_path, img)
            dst = os.path.join(dst_root, img)
            try:
                shutil.copy2(src, dst)  # copy2保留文件元数据，比copy更规范
            except Exception as e:
                print(f"⚠️  复制失败 {img} → {dst}，错误：{e}")

    # 执行复制
    copy_imgs(train_imgs, "train")
    copy_imgs(val_imgs, "val")
    copy_imgs(test_imgs, "test")

    print(
        f"✅ [{cls_name}] 拆分完成：总数={total_imgs} → 训练集={len(train_imgs)} | 验证集={len(val_imgs)} | 测试集={len(test_imgs)}")
    return len(train_imgs), len(val_imgs), len(test_imgs)


def main():
    # 1. 检查原始数据集根目录是否存在
    if not os.path.isdir(RAW_DATA_DIR):
        raise FileNotFoundError(f"❌ 原始数据集目录 {RAW_DATA_DIR} 不存在！请确认路径是否正确")

    # 2. 创建目标目录
    create_target_dirs()

    # 3. 遍历所有类别拆分
    total_train, total_val, total_test = 0, 0, 0
    for cls in CLASS_NAMES:
        t, v, te = split_single_class(cls)
        total_train += t
        total_val += v
        total_test += te

    # 4. 输出总统计
    print("\n" + "=" * 60)
    print(f"📊 整体拆分统计：")
    print(f"   训练集总数：{total_train} 张")
    print(f"   验证集总数：{total_val} 张")
    print(f"   测试集总数：{total_test} 张")
    print(f"   总图片数：{total_train + total_val + total_test} 张")
    print("=" * 60)
    print(f"\n🎉 数据集拆分全部完成！输出目录：{os.path.abspath(TARGET_DATA_DIR)}")


if __name__ == "__main__":
    main()