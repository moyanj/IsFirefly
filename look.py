import argparse
import os
import sys
import torch
from datetime import datetime


def load_model(model_path):
    """安全加载PyTorch模型文件"""
    try:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        model = torch.load(model_path, map_location="cpu")
        return model
    except Exception as e:
        print(f"❌ 加载模型失败: {str(e)}")
        sys.exit(1)


def format_timestamp(timestamp):
    """格式化时间戳为可读格式"""
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "未知时间"


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="PyTorch模型检查点分析工具")
    parser.add_argument("model_path", type=str, help="模型检查点文件路径")
    args = parser.parse_args()

    # 加载模型
    model = load_model(args.model_path)

    # 打印基础信息
    print(f"\n🔍 分析模型: {os.path.basename(args.model_path)}")
    print(f"📁 文件路径: {os.path.abspath(args.model_path)}")
    print(f"🕒 保存时间: {format_timestamp(model.get('time', 0))}")
    print(f"📝 检查点类型: {model.get('type', '未指定')}")

    # 检查可恢复训练状态
    required_keys = {"optimizer_state", "scheduler_state", "model_state"}
    missing_keys = required_keys - set(model.keys())
    resumable = len(missing_keys) == 0

    print(f"\n🔄 可恢复训练: {'✅ 是' if resumable else '❌ 否'}")
    if not resumable and missing_keys:
        print(f"  缺失关键状态: {', '.join(missing_keys)}")

    # 打印训练信息
    print(f"\n📊 训练信息:")
    if "epoch" in model:
        print(f"  - 当前轮次: {model['epoch']}")

    checkpoint_type = model.get("type", "")
    if checkpoint_type == "step":
        print(f"  - 全局步数: {model.get('step', 'N/A')}")
        print(f"  - 训练损失: {model.get('loss', 'N/A'):.6f}")
    elif checkpoint_type == "epoch":
        print(f"  - 测试损失: {model.get('val_loss', 'N/A'):.6f}")
        print(f"  - 测试准确率: {model.get('val_accuracy', 'N/A'):.2f}%")
        print(f"  - 最佳准确率: {model.get('best_accuracy', 'N/A'):.2f}%")
        print(f"  - 全局步数: {model.get('step', 'N/A')}")
        print(f"  - 最终轮次: {'是' if model.get('is_last', False) else '否'}")

    if "args" in model:
        print("\n⚙️ 训练参数:")
        args = model["args"]
        print(f"  - 训练总轮次: {args['epochs']}")
        print(f"  - 批次大小: {args['batch_size']}")
        print(f"  - 学习率: {args['lr']}")
        print(f"  - 数据加载工作线程数: {args['num_workers']}")
        print(f"  - 冻结主干网络权重: {'是' if args['freeze_backbone'] else '否'}")
        print(f"  - 基础模型名称: {args['model_name']}")
        print(f"  - 启用torch.compile()优化: {'是' if args['compile'] else '否'}")
        print(
            f"  - 检查点保存间隔（步数）: {'是' if args['checkpoint_interval'] else '否'}"
        )


if __name__ == "__main__":
    main()
