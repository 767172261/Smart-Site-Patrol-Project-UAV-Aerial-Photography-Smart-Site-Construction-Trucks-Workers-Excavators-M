智慧工地巡检项目-无人机航拍智慧工地施工之卡车、工人、挖掘机移动工程车辆检测数据集


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0e16999435e74463a8e9051b094c2567.png)

1
| 维度       | 详情                                                                                                                                          |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| 数据集类别 | 目标检测类数据集，核心标注类别分为三类：挖掘机（Excavator）、卡车（Trucks）和工作人员（Worker），覆盖工程作业场景中的关键实体对象。          |
| 数据数量   | 包含400张图像数据，配套4个数据集文件，暂未构建可用模型（模型数量为0）。这些图像样本规模可满足基础目标检测模型的训练与验证需求。               |
| 数据格式   | 图像文件通常采用JPG、PNG等通用格式；对应标注文件可能采用XML、TXT等格式记录目标边界框坐标及类别信息，适配主流目标检测算法输入需求。             |
| 核心应用价值 | 1. 支撑工程作业场景下的智能监控系统开发，识别挖掘机、卡车的作业状态与工作人员位置，辅助规避机械与人身碰撞风险。<br>2. 为工程车辆调度算法提供数据支持，通过识别车辆类型与数量优化场地内调度效率。<br>3. 助力施工安全合规检查模型训练，自动检测工作人员是否处于危险作业区域，降低人工巡检成本。 |

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ba4f6ccce2c94d3885ee880465bb9aa0.png)

1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/70f29a025977466694a812780b777dd4.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c0285dcf2aa34f0ea3af3bf1726225ea.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ff3fa83e516243f9a2113618670b3ab9.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/df655906e55c4e3f9918054525745d65.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c561a08fd27c4801867f7465719f7a1d.png)
1
工程场景目标检测数据集（含挖掘机、卡车、工作人员三类）的 完整 YOLOv8 训练代码，基于 ultralytics 框架，支持标准图像格式（JPG/PNG）和 YOLO 格式标注（TXT），适用于训练、验证、推理与部署。

✅ 一、数据集结构要求

bash
construction_dataset/
│
├── images/
│ ├── train/ # 320 张
│ └── val/ # 80 张
│
├── labels/
│ ├── train/ # 对应 TXT 标注文件
│ └── val/
│
├── data.yaml # 数据配置文件
└── README.md
💡 建议按 8:2 划分训练集与验证集（400 × 0.8 = 320）

📄 二、data.yaml 配置文件

yaml
data.yaml
train: ./construction_dataset/images/train
val: ./construction_dataset/images/val
类别数量与名称
nc: 3
names: ['Excavator', 'Truck', 'Worker']
⚠️ 注意：类别顺序需与标注文件中的类别索引一致（0=挖掘机, 1=卡车, 2=工作人员）

🛠 三、YOLOv8 训练代码（详细版）
✅ train_construction.py

python
train_construction.py
from ultralytics import YOLO
import torch
import os
==================== 配置参数 ====================
MODEL_TYPE = 'yolov8s.pt' # 可选: yolov8n, yolov8m, yolov8l
DATA_CONFIG = 'construction_dataset/data.yaml'
EPOCHS = 150
BATCH_SIZE = 32
IMG_SIZE = 640 # 可根据显存调整为 320, 416, 512
PROJECT_NAME = 'construction_detection'
EXPERIMENT_NAME = 'yolov8s_640_construction'
模型保存路径
os.makedirs(f"runs/train/{EXPERIMENT_NAME}", exist_ok=True)
==================== 设备选择 ====================
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
print(f"✅ 使用设备: {DEVICE}")
==================== 加载预训练模型 ====================
model = YOLO(MODEL_TYPE) # 自动下载或加载本地 .pt 文件
==================== 开始训练 ====================
results = model.train(
data=DATA_CONFIG,
epochs=EPOCHS,
batch=BATCH_SIZE,
imgsz=IMG_SIZE,
project=PROJECT_NAME,
name=EXPERIMENT_NAME,
save=True,
save_period=10, # 每10轮保存一次
device=DEVICE,
workers=4, # 数据加载线程数
optimizer='AdamW', # 推荐用于小数据集
lr0=1e-3, # 初始学习率
lrf=1e-2, # 最终学习率 = lr0 * lrf
momentum=0.937,
weight_decay=1e-4,
warmup_epochs=3.0, # 学习率预热
box=7.5, # 损失权重
cls=0.5,
dfl=1.5,
close_mosaic=10, # 关闭 Mosaic 增强的轮次
# -------------------------------
# 数据增强参数（适合小数据集）
# -------------------------------
augment=True, # 启用增强
hsv_h=0.015, # 色调抖动
hsv_s=0.7, # 饱和度抖动
hsv_v=0.4, # 明度抖动
degrees=15.0, # 随机旋转
translate=0.3, # 平移
scale=0.7, # 缩放
shear=2.0, # 剪切
flipud=0.5, # 上下翻转
fliplr=0.5, # 左右翻转
mosaic=1.0, # Mosaic 增强概率
mixup=0.1, # MixUp 增强
copy_paste=0.1, # Copy-Paste 增强（小目标有效）
patience=20, # 早停轮数
)

print("✅ 训练完成！")

🔍 四、验证与评估

python
validate.py
from ultralytics import YOLO
加载最佳模型
model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')
验证
metrics = model.val(
data=DATA_CONFIG,
split='val',
imgsz=IMG_SIZE,
batch=BATCH_SIZE,
conf=0.001,
iou=0.6,
device=DEVICE,
save_json=True, # 生成 COCO 格式结果（可选）
plots=True, # 生成 PR 曲线、混淆矩阵等
)
输出关键指标
print(f"📊 mAP@0.5: {metrics.box.map:.4f}")
print(f"📊 mAP@0.5:0.95: {metrics.box.map50:.4f}")
print(f"📊 Precision: {metrics.box.p:.4f}")
print(f"📊 Recall: {metrics.box.r:.4f}")

🔦 五、推理代码（单图/视频/文件夹）

python
predict.py
from ultralytics import YOLO

model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')
支持多种输入源
source = 'construction_dataset/images/val' # 或单张图、视频、摄像头
results = model.predict(
source=source,
imgsz=IMG_SIZE,
conf=0.3,
iou=0.5,
device=DEVICE,
show=False,
save=True,
save_conf=True, # 保存置信度
project='runs/predict/construction',
name='result',
line_width=2,
classes=None, # 可指定类别 [0, 1, 2]
)

for r in results:
boxes = r.boxes
print(f"🖼️ 图像: {r.path}")
for box in boxes:
cls = int(box.cls)
conf = float(box.conf)
print(f" 🎯 {model.names[cls]}: {conf:.2f}")

📦 六、模型导出（ONNX / TensorRT / TFLite）

python
export_model.py
from ultralytics import YOLO

model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')
导出为 ONNX（通用部署）
model.export(
format='onnx',
opset=12,
dynamic=True,
simplify=True,
input_shape=[1, 3, 640, 640]
)
可选：导出为 TensorRT（NVIDIA GPU 加速）
model.export(format='engine', device=0, half=True)
可选：导出为 TFLite（移动端）
model.export(format='tflite')

🎯 七、优化建议（针对 400 张小数据集）

方法 说明
------ ------
数据增强增强 使用 mixup, copy_paste 提升小样本泛化能力
预训练权重 使用 yolov8m.pt 等大模型 + 预训练权重
早停（Early Stop） 设置 patience=20 防止过拟合
学习率预热 warmup_epochs=3 提升训练稳定性
高分辨率训练 若显存允许，使用 imgsz=896 提升小目标检出率


