# IsACG - ACG风格图像分类模型

<p align="center">
  <a href="https://huggingface.co/moyanjdc/IsACG"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.7-red.svg"></a>
  <a href="https://github.com/moyanjdc/IsACG/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="https://github.com/moyanjdc/IsACG"><img src="https://img.shields.io/badge/GitHub-Repository-black"></a>
</p>

IsACG是一个轻量级的二分类图像分类模型系列，专门用于判断图像是否为ACG（动画、漫画、游戏）或二次元风格。项目基于PyTorch实现，提供完整的训练、评估和部署流程。

## 🌟 模型特色

- **🎯 高精度**: 在ACG识别任务上达到99%准确率
- **⚡ 轻量快速**: 参数量仅2.5M-5.5M，推理速度快
- **🔄 多格式支持**: 原生PyTorch、ONNX格式，便于部署
- **📱 多端适用**: 支持CPU、GPU、移动端部署
- **🔗 完整工具链**: 包含数据预处理、训练、评估、转换和部署的全套工具

## 模型版本

| 版本 | 架构              | 参数 | 准确率 | 特点                   |
| ---- | ----------------- | ---- | ------ | ---------------------- |
| v1   | MobileNetV3-Large | 5.5M | ~99.1% | 高精度，适合服务器部署 |
| v1s  | MobileNetV3-Small | 2.5M | ~98.9% | 轻量快速，适合移动端   |
| v2   | MobileNetV3-Small | 2.5M | ~97.5% | 改进泛化能力           |

## 🎯 使用场景

### ✅ 推荐使用
- 动漫/漫画内容过滤
- 游戏截图识别
- 二次元风格检测
- 内容审核系统
- 图像分类管道

### ⚠️ 注意事项
- 主要针对风格识别，而非内容理解
- 对于高度风格化的图像（如3D渲染动漫）可能误判
- 建议图像分辨率不低于256×256

## 快速开始

### 安装依赖

```bash
uv sync
```

### 使用预训练模型

1. **下载模型**：
   - 从[Hugging Face仓库](https://huggingface.co/moyanjdc/IsACG/)下载预训练模型
   - 将模型文件放入`models/release/`目录

2. **命令行推理**：
```bash
python isacg.py --model_path models/release/IsACG_v1s_98.94%.onnx --image_path your_image.jpg
```

3. **Web界面**：
```bash
python webapp.py
```
访问 http://localhost:8080 使用图形界面

4. **API服务**：
```bash
python onnx_server.py
```
使用POST请求调用 `/predict` 接口

## 完整使用流程

### 1. 数据准备

```bash
# 准备原始数据
# 正样本（ACG风格）放在 dataset/yes/
# 负样本（非ACG风格）放在 dataset/no/

# 运行预处理
python pre_process.py
```

### 2. 训练模型

```bash
# 基本训练
python train.py --model_name mobilenet_v3_small --epochs 20 --batch_size 32

# 使用预训练权重
python train.py --model_name mobilenet_v3_small --use_pretrained

# 启用TensorBoard监控
python train.py --model_name mobilenet_v3_large --use_tensorboard
```

### 3. 模型转换与优化

```bash
# 转换为ONNX格式
python conv.py checkpoint.pth -v 1s -a 98.94

# 模型量化（减少模型大小）
python qua.py
```

### 4. 模型分析

```bash
# 查看模型检查点信息
python look.py checkpoint.pth
```

## 项目结构

```
IsFirefly/
├── model.py              # 模型定义
├── train.py              # 训练脚本
├── dataset.py            # 数据加载
├── isacg.py              # 推理评估
├── conv.py               # 模型转换（PyTorch→ONNX）
├── qua.py                # 模型量化
├── webapp.py             # Gradio Web界面
├── onnx_server.py        # Flask API服务
├── pre_process.py        # 数据预处理
├── make_unplash.py       # Unplash数据下载工具
├── look.py               # 模型检查点分析
├── pyproject.toml        # 项目配置
└── README.md             # 说明文档
```

## API接口说明

### REST API (onnx_server.py)

```python
# 请求示例
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict

# 响应格式
{
    "predicted_class": "acg",
    "confidence": 0.9945,
    "success": true
}
```

### Python API

```python
from isacg import TorchInference, ONNXInference
from PIL import Image

# 使用PyTorch模型
predictor = TorchInference("model.pth", device="cuda")
result = predictor.predict(Image.open("image.jpg"))

# 使用ONNX模型
predictor = ONNXInference("model.onnx", device="cuda")
result = predictor.predict("image.jpg")
```

## 训练配置

### 超参数
- 输入尺寸：512×512
- 学习率：1e-3
- 优化器：Adam
- 损失函数：交叉熵损失
- 学习率调度：ReduceLROnPlateau
- 数据增强：随机旋转、水平翻转、日晒效果

### 数据增强
项目包含多种数据增强技术，提高模型泛化能力：
- 随机旋转 (±25度)
- 随机水平翻转 (p=0.5)
- 随机日晒效果 (p=0.5)

### 性能指标

| 设备            | 推理速度     | 内存占用 | 推荐用途 |
| --------------- | ------------ | -------- | -------- |
| CPU (E5-2673V3) | ~25FPS  | ~200MB   | 本地测试 |
| GPU (GTX 750)  | ~40FPS   | ~500MB   | 生产部署 |

## 📝 技术细节

### 模型架构
- **基础网络**: MobileNetV3-Large/Small
- **输入尺寸**: 512×512
- **输出**: 二分类（0: 非ACG, 1: ACG）
- **参数量**: 2.5M (v1s/v2) / 5.5M (v1)

### 训练数据
- **正样本**: 50,00+ ACG图像（动漫、漫画、游戏）（来自@Scighost/爬虫）
- **负样本**: 50,00+ 真实照片、插画等（来自Unplash）
- **数据增强**: 随机旋转、翻转、色彩调整


## 贡献指南

欢迎贡献代码、数据或文档改进：

1. Fork本仓库
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 致谢

- 感谢PyTorch和TorchVision团队
- 特别感谢[@Scighost](https://github.com/Scighost)老师的铯图（25%的训练集）
- 感谢[Unplash](https://unsplash.com/)提供的真实图片图片数据集
- 感谢开源社区的支持

## 联系方式

- 项目主页：https://github.com/moyanj/IsACG
- 问题反馈：GitHub Issues
- 模型下载：[Hugging Face Hub](https://huggingface.co/moyanjdc/IsACG/)

---

*注：本项目主要用于教育和研究目的。商业使用请确保遵守相关法律法规和版权要求。*

<p align="center">
  Made with ❤️ by <a href="https://github.com/moyanjdc">moyanjdc</a> · 
  <a href="https://huggingface.co/moyanjdc/IsACG">HF Hub</a> · 
  <a href="https://github.com/moyanjdc/IsACG">GitHub</a>
</p>