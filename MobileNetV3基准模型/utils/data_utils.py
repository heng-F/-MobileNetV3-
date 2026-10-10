from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# 数据预处理（MobileNetV3官方推荐）
def get_data_transforms(image_size=(224, 224)):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                             std=[0.229, 0.224, 0.225])  # ImageNet方差
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# 加载数据集
def get_data_loaders(train_dir, val_dir, batch_size, image_size):
    train_transform, val_transform = get_data_transforms(image_size)

    # 加载训练集（按文件夹自动分类）
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    # 加载验证集
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader