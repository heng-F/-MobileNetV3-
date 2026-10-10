# plot_comparison.py
import json
import os
import matplotlib.pyplot as plt

# 根据你的实际路径修改此字典（模型名称: history.json 的完整路径）
# 这里使用项目根目录下的 output 子目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_HISTORY = {
    'MobileNetV3-Small': os.path.join(ROOT_DIR, 'output', 'mobilenetv3_small', 'history.json'),
    'EfficientNet-B0':   os.path.join(ROOT_DIR, 'output', 'efficientnetb0', 'history.json'),
    'ShuffleNet V2':     os.path.join(ROOT_DIR, 'output', 'shufflenetv2', 'history.json'),
    'SqueezeNet':        os.path.join(ROOT_DIR, 'output', 'squeezenet', 'history.json'),
}

def load_history(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_curves(data_dict, y_label, title, save_path, y_lim=None):
    plt.figure(figsize=(10, 6))
    for name, hist in data_dict.items():
        # 使用验证集长度作为 epoch 数（所有记录长度一致）
        epochs = range(1, len(hist['val_losses']) + 1)
        if 'Train' in title:
            y_values = hist['train_losses']
        elif 'Validation Loss' in title:
            y_values = hist['val_losses']
        else:  # Validation Accuracy
            y_values = hist['val_accs']
        plt.plot(epochs, y_values, marker='o', markersize=3, linewidth=1.5, label=name)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    if y_lim:
        plt.ylim(y_lim)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Saved: {save_path}')

def main():
    all_data = {}
    for name, path in MODEL_HISTORY.items():
        if os.path.exists(path):
            all_data[name] = load_history(path)
            print(f'Loaded {name} from {path} (epochs={len(all_data[name]["val_losses"])})')
        else:
            print(f'Warning: {path} not found, skip {name}')

    if not all_data:
        print('No history data found. Please train models first.')
        return

    # 1. 训练损失对比
    plot_curves(all_data, 'Loss', 'Training Loss Comparison',
                'train_loss_comparison.png', y_lim=None)

    # 2. 验证损失对比
    plot_curves(all_data, 'Loss', 'Validation Loss Comparison',
                'val_loss_comparison.png', y_lim=None)

    # 3. 验证准确率对比
    plot_curves(all_data, 'Accuracy (%)', 'Validation Accuracy Comparison',
                'val_acc_comparison.png', y_lim=(0, 100))

    print('All comparison figures generated.')

if __name__ == '__main__':
    main()