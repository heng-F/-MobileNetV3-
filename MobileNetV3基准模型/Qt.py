# import sys
# import os
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
#     QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit, QGroupBox,
#     QMessageBox, QSizePolicy, QSpacerItem, QFrame
# )
# from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
# from PIL import Image
# import numpy as np
#
# # 导入项目配置和模型
# sys.path.append(r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型")
# import config
# from models.mobilenetv3_small import build_small_model
#
#
# # ======================== 性能优化：预测线程（避免UI卡顿） ========================
# class PredictThread(QThread):
#     """独立线程执行预测，避免UI阻塞"""
#     result_signal = pyqtSignal(str)  # 传递预测结果
#     error_signal = pyqtSignal(str)  # 传递错误信息
#     status_signal = pyqtSignal(str)  # 传递状态信息
#
#     def __init__(self, model, image_path, transform, device, class_names):
#         super().__init__()
#         self.model = model
#         self.image_path = image_path
#         self.transform = transform
#         self.device = device
#         self.class_names = class_names
#
#     def run(self):
#         try:
#             self.status_signal.emit("正在预处理图像...")
#             # 图像预处理（优化IO和计算）
#             image = Image.open(self.image_path).convert('RGB')
#             input_tensor = self.transform(image).unsqueeze(0).to(self.device, non_blocking=True)
#
#             self.status_signal.emit("正在预测...")
#             # 预测（性能优化：禁用梯度、设置eval模式、非阻塞传输）
#             with torch.no_grad():
#                 torch.cuda.empty_cache()  # 清理显存
#                 outputs = self.model(input_tensor)
#                 probabilities = F.softmax(outputs, dim=1)
#                 _, predicted_idx = torch.max(outputs, 1)
#                 predicted_class = self.class_names[predicted_idx.item()]
#                 confidence = probabilities[0][predicted_idx.item()].item() * 100
#
#             # 生成结果（优化排序，只显示前5个类别）
#             result_lines = ["=" * 50, "📌 分类结果", "=" * 50]
#             result_lines.append(f"预测类别：{predicted_class}")
#             result_lines.append(f"置信度：{confidence:.2f}%")
#             result_lines.append("\n🔥 前5高置信度类别：")
#
#             # 按置信度排序，只显示前5个（减少计算和渲染耗时）
#             prob_sorted = torch.argsort(probabilities[0], descending=True)
#             for i in range(min(5, len(self.class_names))):
#                 idx = prob_sorted[i]
#                 class_name = self.class_names[idx]
#                 prob = probabilities[0][idx].item() * 100
#                 result_lines.append(f"{class_name}: {prob:.2f}%")
#
#             self.result_signal.emit("\n".join(result_lines))
#             self.status_signal.emit("分类完成！")
#
#         except Exception as e:
#             self.error_signal.emit(str(e))
#             self.status_signal.emit("预测失败")
#
#
# # ======================== 主界面（美化+性能优化） ========================
# class GarbageClassificationUI(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.model = None
#         self.predict_thread = None
#         self.device = torch.device(config.DEVICE)
#         self.init_optimized_transform()  # 优化预处理
#         self.init_ui()  # 初始化美化后的UI
#
#     def init_optimized_transform(self):
#         """优化图像预处理（减少计算量，提升速度）"""
#         self.transform = transforms.Compose([
#             transforms.Resize(256, transforms.InterpolationMode.BILINEAR),  # 更快的插值方式
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#
#     def init_ui(self):
#         """初始化美化后的界面"""
#         # 基础窗口设置
#         self.setWindowTitle("📱 基于MobileNetV3的垃圾图像分类系统")
#         self.setGeometry(150, 100, 1100, 750)
#         self.setStyleSheet("""
#             QMainWindow {
#                 background-color: #f5f7fa;
#             }
#             QGroupBox {
#                 font-size: 14px;
#                 font-weight: bold;
#                 color: #2c3e50;
#                 border: 2px solid #e1e8ed;
#                 border-radius: 8px;
#                 margin-top: 10px;
#                 padding-top: 10px;
#             }
#             QGroupBox::title {
#                 subcontrol-origin: margin;
#                 left: 10px;
#                 padding: 0 10px 0 10px;
#             }
#             QPushButton {
#                 background-color: #3498db;
#                 color: white;
#                 border: none;
#                 border-radius: 6px;
#                 padding: 8px 16px;
#                 font-size: 13px;
#                 font-weight: 500;
#             }
#             QPushButton:hover {
#                 background-color: #2980b9;
#             }
#             QPushButton:disabled {
#                 background-color: #bdc3c7;
#                 color: #7f8c8d;
#             }
#             QLineEdit {
#                 border: 1px solid #ddd;
#                 border-radius: 6px;
#                 padding: 8px;
#                 font-size: 13px;
#                 background-color: white;
#             }
#             QTextEdit {
#                 border: 1px solid #ddd;
#                 border-radius: 6px;
#                 padding: 10px;
#                 font-size: 13px;
#                 background-color: white;
#                 font-family: Consolas, monospace;
#             }
#             QLabel {
#                 font-size: 13px;
#                 color: #2c3e50;
#             }
#             QStatusBar {
#                 font-size: 12px;
#                 color: #7f8c8d;
#             }
#         """)
#
#         # 字体设置
#         font = QFont("Microsoft YaHei", 9)
#         self.setFont(font)
#
#         # 中心部件
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
#
#         # 主布局（带间距）
#         main_layout = QVBoxLayout(central_widget)
#         main_layout.setSpacing(20)
#         main_layout.setContentsMargins(30, 30, 30, 20)
#
#         # 1. 标题栏
#         title_label = QLabel("🗑️ 垃圾图像分类系统（MobileNetV3蒸馏版）")
#         title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
#         title_label.setAlignment(Qt.AlignCenter)
#         title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
#         main_layout.addWidget(title_label)
#
#         # 2. 模型配置区域（美化+紧凑）
#         model_group = QGroupBox("⚙️ 模型配置")
#         model_layout = QHBoxLayout(model_group)
#         model_layout.setSpacing(15)
#         model_layout.setContentsMargins(20, 15, 20, 15)
#
#         # 模型路径输入
#         self.model_path_edit = QLineEdit()
#         default_model_path = r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型\output\distilled\student_distilled.pth"
#         self.model_path_edit.setText(default_model_path)
#         self.model_path_edit.setMinimumWidth(500)
#
#         # 按钮
#         browse_btn = QPushButton("📂 浏览")
#         browse_btn.clicked.connect(self.browse_model_path)
#
#         load_model_btn = QPushButton("🚀 加载模型")
#         load_model_btn.clicked.connect(self.load_model)
#         load_model_btn.setStyleSheet("background-color: #2ecc71;")
#
#         model_layout.addWidget(QLabel("模型路径："))
#         model_layout.addWidget(self.model_path_edit)
#         model_layout.addWidget(browse_btn)
#         model_layout.addWidget(load_model_btn)
#
#         main_layout.addWidget(model_group)
#
#         # 3. 图像和结果区域（分栏美化）
#         content_layout = QHBoxLayout()
#         content_layout.setSpacing(20)
#
#         # 左侧：图像显示区
#         image_frame = QFrame()
#         image_frame.setStyleSheet("""
#             QFrame {
#                 border: 2px solid #e1e8ed;
#                 border-radius: 10px;
#                 background-color: white;
#             }
#         """)
#         image_layout = QVBoxLayout(image_frame)
#         image_layout.setSpacing(15)
#         image_layout.setContentsMargins(20, 20, 20, 20)
#
#         # 图像选择按钮
#         select_image_btn = QPushButton("🖼️ 选择图像")
#         select_image_btn.setStyleSheet("background-color: #9b59b6;")
#         select_image_btn.clicked.connect(self.select_image)
#         image_layout.addWidget(select_image_btn, alignment=Qt.AlignCenter)
#
#         # 图像显示标签（圆角+阴影）
#         self.image_label = QLabel()
#         self.image_label.setFixedSize(450, 450)
#         self.image_label.setStyleSheet("""
#             QLabel {
#                 border: 2px dashed #bdc3c7;
#                 border-radius: 8px;
#                 background-color: #f8f9fa;
#             }
#         """)
#         self.image_label.setAlignment(Qt.AlignCenter)
#         self.image_label.setText("请选择图像\n（支持jpg/png/bmp）")
#         image_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
#
#         # 右侧：结果展示区
#         result_frame = QFrame()
#         result_frame.setStyleSheet("""
#             QFrame {
#                 border: 2px solid #e1e8ed;
#                 border-radius: 10px;
#                 background-color: white;
#             }
#         """)
#         result_layout = QVBoxLayout(result_frame)
#         result_layout.setSpacing(15)
#         result_layout.setContentsMargins(20, 20, 20, 20)
#
#         # 预测按钮
#         self.predict_btn = QPushButton("🔍 开始分类")
#         self.predict_btn.setStyleSheet("background-color: #e67e22;")
#         self.predict_btn.clicked.connect(self.start_predict)
#         self.predict_btn.setEnabled(False)
#         result_layout.addWidget(self.predict_btn, alignment=Qt.AlignCenter)
#
#         # 结果文本框（美化+自适应）
#         self.result_text = QTextEdit()
#         self.result_text.setReadOnly(True)
#         self.result_text.setPlaceholderText("分类结果将显示在这里...")
#         self.result_text.setMinimumHeight(380)
#         result_layout.addWidget(self.result_text)
#
#         # 添加到内容布局
#         content_layout.addWidget(image_frame)
#         content_layout.addWidget(result_frame)
#         main_layout.addLayout(content_layout)
#
#         # 状态栏（美化）
#         self.status_bar = self.statusBar()
#         self.status_bar.setStyleSheet("QStatusBar { background-color: #f5f7fa; border-top: 1px solid #e1e8ed; }")
#         self.status_bar.showMessage("✅ 就绪 - 请先加载模型", 5000)
#
#     # ======================== 核心功能函数（优化+复用） ========================
#     def browse_model_path(self):
#         """浏览选择模型文件"""
#         file_path, _ = QFileDialog.getOpenFileName(
#             self, "选择模型权重文件", "", "PyTorch模型文件 (*.pth *.pt)"
#         )
#         if file_path:
#             self.model_path_edit.setText(file_path)
#
#     def load_model(self):
#         """优化模型加载速度（预加载+设备优化）"""
#         model_path = self.model_path_edit.text().strip()
#         if not model_path or not os.path.exists(model_path):
#             QMessageBox.warning(self, "⚠️ 警告", "模型文件路径无效！")
#             return
#
#         try:
#             self.status_bar.showMessage("🔄 正在加载模型...")
#             # 优化：提前清理显存
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#
#             # 加载模型（复用config配置，减少硬编码）
#             self.model = build_small_model(num_classes=config.NUM_CLASSES, pretrained=False)
#             # 优化：加载时指定设备，减少后续传输
#             state_dict = torch.load(model_path, map_location=self.device)
#             self.model.load_state_dict(state_dict)
#             self.model.to(self.device)
#             self.model.eval()  # 提前设置eval模式
#
#             self.status_bar.showMessage("✅ 模型加载成功！", 3000)
#             QMessageBox.information(self, "🎉 成功", "蒸馏模型加载完成！")
#             self.predict_btn.setEnabled(True)
#
#         except Exception as e:
#             QMessageBox.critical(self, "❌ 错误", f"模型加载失败：{str(e)}")
#             self.status_bar.showMessage("❌ 模型加载失败", 5000)
#
#     def select_image(self):
#         """优化图像加载（异步+压缩）"""
#         image_path, _ = QFileDialog.getOpenFileName(
#             self, "选择图像文件", "", "图像文件 (*.jpg *.jpeg *.png *.bmp)"
#         )
#         if image_path:
#             try:
#                 # 优化：缩放图像时使用快速插值，减少内存占用
#                 pixmap = QPixmap(image_path)
#                 pixmap = pixmap.scaled(
#                     450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation
#                 )
#                 self.image_label.setPixmap(pixmap)
#                 self.current_image_path = image_path
#                 self.result_text.clear()
#                 self.status_bar.showMessage(f"🖼️ 已加载图像：{os.path.basename(image_path)}", 3000)
#
#             except Exception as e:
#                 QMessageBox.critical(self, "❌ 错误", f"图像加载失败：{str(e)}")
#
#     def start_predict(self):
#         """启动独立线程预测，避免UI卡顿"""
#         if not hasattr(self, 'current_image_path') or self.model is None:
#             QMessageBox.warning(self, "⚠️ 警告", "请先选择图像并加载模型！")
#             return
#
#         # 禁用预测按钮，防止重复点击
#         self.predict_btn.setEnabled(False)
#         self.result_text.clear()
#
#         # 创建预测线程（核心性能优化）
#         self.predict_thread = PredictThread(
#             model=self.model,
#             image_path=self.current_image_path,
#             transform=self.transform,
#             device=self.device,
#             class_names=config.CLASS_NAMES
#         )
#
#         # 绑定线程信号
#         self.predict_thread.result_signal.connect(self.show_predict_result)
#         self.predict_thread.error_signal.connect(self.show_predict_error)
#         self.predict_thread.status_signal.connect(self.status_bar.showMessage)
#
#         # 线程结束后重新启用按钮
#         self.predict_thread.finished.connect(lambda: self.predict_btn.setEnabled(True))
#
#         # 启动线程
#         self.predict_thread.start()
#
#     def show_predict_result(self, result):
#         """显示预测结果"""
#         self.result_text.setText(result)
#         self.status_bar.showMessage("✅ 分类完成！", 3000)
#
#     def show_predict_error(self, error):
#         """显示预测错误"""
#         QMessageBox.critical(self, "❌ 预测错误", f"分类失败：{error}")
#         self.status_bar.showMessage("❌ 预测失败", 5000)
#
#
# # ======================== 程序入口 ========================
# if __name__ == "__main__":
#     # 全局性能优化：启用PyTorch异步执行
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True  # 针对固定尺寸图像加速
#
#     app = QApplication(sys.argv)
#     # 美化应用程序外观
#     app.setStyle("Fusion")  # 统一跨平台样式
#     palette = QPalette()
#     palette.setColor(QPalette.Window, QColor(245, 247, 250))
#     app.setPalette(palette)
#
#     window = GarbageClassificationUI()
#     window.show()
#     sys.exit(app.exec_())


import sys
import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit, QGroupBox,
    QMessageBox, QSizePolicy, QSpacerItem, QFrame, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image

# 导入项目配置（CLASS_NAMES, NUM_CLASSES等）
sys.path.append(r"D:\pycharm实验\基于MobileNetV3的垃圾图像分类系统\MobileNetV3基准模型")
import config


# ======================== 模型构建工厂 ========================
def build_model_by_type(model_type, num_classes, device):
    """根据模型类型构建网络结构（未加载权重）"""
    if model_type == "EfficientNet-B0":
        model = models.efficientnet_b0(pretrained=False)
        # 替换分类头
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    elif model_type == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_type == "SqueezeNet":
        model = models.squeezenet1_0(pretrained=False)
        # SqueezeNet 的分类器是 nn.Conv2d，需要替换最后一层卷积的输出通道数
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif model_type == "MobileNetV3-Small":
        from models.mobilenetv3_small import build_small_model
        model = build_small_model(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model = model.to(device)
    model.eval()
    return model


def load_model_weights(model, weight_path, device):
    """加载权重文件，自动处理键名前缀（如 module.）和严格性"""
    state_dict = torch.load(weight_path, map_location=device)

    # 如果保存的是整个 checkpoint 字典（包含 model_state_dict 键），则提取
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # 移除可能存在的 'module.' 前缀（多GPU训练保存的权重）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # 严格加载，若失败则尝试非严格模式
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("权重加载成功（严格模式）")
    except Exception as e:
        print(f"严格模式加载失败，尝试非严格模式: {e}")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"缺失键: {missing[:5] if len(missing) > 5 else missing}")
        print(f"多余键: {unexpected[:5] if len(unexpected) > 5 else unexpected}")

    return model


# ======================== 预测线程 ========================
class PredictThread(QThread):
    result_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self, model, image_path, transform, device, class_names):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.transform = transform
        self.device = device
        self.class_names = class_names

    def run(self):
        try:
            self.status_signal.emit("正在预处理图像...")
            image = Image.open(self.image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            self.status_signal.emit("正在预测...")
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_class = self.class_names[predicted_idx.item()]
                confidence = probabilities[0][predicted_idx.item()].item() * 100

            result_lines = ["=" * 50, "📌 分类结果", "=" * 50]
            result_lines.append(f"预测类别：{predicted_class}")
            result_lines.append(f"置信度：{confidence:.2f}%")
            result_lines.append("\n🔥 前5高置信度类别：")

            prob_sorted = torch.argsort(probabilities[0], descending=True)
            for i in range(min(5, len(self.class_names))):
                idx = prob_sorted[i]
                class_name = self.class_names[idx]
                prob = probabilities[0][idx].item() * 100
                result_lines.append(f"{class_name}: {prob:.2f}%")

            self.result_signal.emit("\n".join(result_lines))
            self.status_signal.emit("分类完成！")
        except Exception as e:
            self.error_signal.emit(str(e))
            self.status_signal.emit("预测失败")


# ======================== 主界面 ========================
class GarbageClassificationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.predict_thread = None
        self.device = torch.device(
            config.DEVICE if hasattr(config, 'DEVICE') else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.current_image_path = None
        self.transform = self._get_transform()
        self.init_ui()

    def _get_transform(self):
        """统一的预处理：缩放至256，中心裁剪224，归一化（ImageNet标准）"""
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def init_ui(self):
        self.setWindowTitle("🗑️ 多架构垃圾图像分类系统 (EfficientNet/ShuffleNet/SqueezeNet)")
        self.setGeometry(150, 100, 1100, 750)
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f7fa; }
            QGroupBox {
                font-size: 14px; font-weight: bold; color: #2c3e50;
                border: 2px solid #e1e8ed; border-radius: 8px;
                margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 10px; }
            QPushButton {
                background-color: #3498db; color: white; border: none;
                border-radius: 6px; padding: 8px 16px; font-size: 13px;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }
            QLineEdit, QComboBox {
                border: 1px solid #ddd; border-radius: 6px; padding: 8px;
                font-size: 13px; background-color: white;
            }
            QTextEdit {
                border: 1px solid #ddd; border-radius: 6px; padding: 10px;
                font-size: 13px; background-color: white;
                font-family: Consolas, monospace;
            }
            QLabel { font-size: 13px; color: #2c3e50; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 20)

        # 标题
        title_label = QLabel("🗑️ 多架构垃圾图像分类系统")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        # 模型配置区域：选择类型 + 路径 + 加载按钮
        model_group = QGroupBox("⚙️ 模型配置")
        model_layout = QHBoxLayout(model_group)
        model_layout.setSpacing(15)
        model_layout.setContentsMargins(20, 15, 20, 15)

        model_layout.addWidget(QLabel("模型架构："))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["EfficientNet-B0", "ShuffleNetV2", "SqueezeNet", "MobileNetV3-Small"])
        model_layout.addWidget(self.model_combo)

        model_layout.addWidget(QLabel("权重文件："))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setMinimumWidth(400)
        model_layout.addWidget(self.model_path_edit)

        browse_btn = QPushButton("📂 浏览")
        browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_btn)

        load_btn = QPushButton("🚀 加载模型")
        load_btn.clicked.connect(self.load_model)
        load_btn.setStyleSheet("background-color: #2ecc71;")
        model_layout.addWidget(load_btn)

        main_layout.addWidget(model_group)

        # 图像和结果区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # 左侧：图像显示
        image_frame = QFrame()
        image_frame.setStyleSheet("border: 2px solid #e1e8ed; border-radius: 10px; background-color: white;")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setSpacing(15)
        image_layout.setContentsMargins(20, 20, 20, 20)

        select_img_btn = QPushButton("🖼️ 选择图像")
        select_img_btn.setStyleSheet("background-color: #9b59b6;")
        select_img_btn.clicked.connect(self.select_image)
        image_layout.addWidget(select_img_btn, alignment=Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setFixedSize(450, 450)
        self.image_label.setStyleSheet("border: 2px dashed #bdc3c7; border-radius: 8px; background-color: #f8f9fa;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("请选择图像\n（支持jpg/png/bmp）")
        image_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # 右侧：结果展示
        result_frame = QFrame()
        result_frame.setStyleSheet("border: 2px solid #e1e8ed; border-radius: 10px; background-color: white;")
        result_layout = QVBoxLayout(result_frame)
        result_layout.setSpacing(15)
        result_layout.setContentsMargins(20, 20, 20, 20)

        self.predict_btn = QPushButton("🔍 开始分类")
        self.predict_btn.setStyleSheet("background-color: #e67e22;")
        self.predict_btn.clicked.connect(self.start_predict)
        self.predict_btn.setEnabled(False)
        result_layout.addWidget(self.predict_btn, alignment=Qt.AlignCenter)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("分类结果将显示在这里...")
        self.result_text.setMinimumHeight(380)
        result_layout.addWidget(self.result_text)

        content_layout.addWidget(image_frame)
        content_layout.addWidget(result_frame)
        main_layout.addLayout(content_layout)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("QStatusBar { background-color: #f5f7fa; border-top: 1px solid #e1e8ed; }")
        self.status_bar.showMessage("✅ 就绪 - 请选择模型类型和权重文件，然后点击加载", 5000)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重文件", "", "PyTorch模型文件 (*.pth *.pt)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def load_model(self):
        model_path = self.model_path_edit.text().strip()
        model_type = self.model_combo.currentText()

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "⚠️ 警告", "权重文件路径无效！")
            return

        try:
            self.status_bar.showMessage(f"🔄 正在加载 {model_type} 模型...")
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 构建模型
            self.model = build_model_by_type(model_type, config.NUM_CLASSES, self.device)
            # 加载权重
            self.model = load_model_weights(self.model, model_path, self.device)
            self.model.eval()

            self.status_bar.showMessage(f"✅ {model_type} 模型加载成功！", 3000)
            QMessageBox.information(self, "🎉 成功", f"{model_type} 模型加载完成！")
            self.predict_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "❌ 错误", f"模型加载失败：\n{str(e)}")
            self.status_bar.showMessage("❌ 模型加载失败", 5000)
            self.model = None
            self.predict_btn.setEnabled(False)

    def select_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if image_path:
            try:
                pixmap = QPixmap(image_path)
                pixmap = pixmap.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
                self.current_image_path = image_path
                self.result_text.clear()
                self.status_bar.showMessage(f"🖼️ 已加载图像：{os.path.basename(image_path)}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "❌ 错误", f"图像加载失败：{str(e)}")

    def start_predict(self):
        if not hasattr(self, 'current_image_path') or self.model is None:
            QMessageBox.warning(self, "⚠️ 警告", "请先选择图像并加载模型！")
            return

        self.predict_btn.setEnabled(False)
        self.result_text.clear()

        self.predict_thread = PredictThread(
            model=self.model,
            image_path=self.current_image_path,
            transform=self.transform,
            device=self.device,
            class_names=config.CLASS_NAMES
        )
        self.predict_thread.result_signal.connect(self.show_predict_result)
        self.predict_thread.error_signal.connect(self.show_predict_error)
        self.predict_thread.status_signal.connect(self.status_bar.showMessage)
        self.predict_thread.finished.connect(lambda: self.predict_btn.setEnabled(True))
        self.predict_thread.start()

    def show_predict_result(self, result):
        self.result_text.setText(result)
        self.status_bar.showMessage("✅ 分类完成！", 3000)

    def show_predict_error(self, error):
        QMessageBox.critical(self, "❌ 预测错误", f"分类失败：{error}")
        self.status_bar.showMessage("❌ 预测失败", 5000)


# ======================== 入口 ========================
if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 247, 250))
    app.setPalette(palette)

    window = GarbageClassificationUI()
    window.show()
    sys.exit(app.exec_())