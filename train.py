"""
深度学习模型训练脚本
支持功能：多模型训练、TensorBoard可视化、断点续训、自动混合精度等
"""

from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import Model  # 自定义模型类
from dataset import dataset  # 自定义数据集类
from tqdm import tqdm  # 进度条工具
import os
import shutil
from datetime import datetime
import numpy as np
import json
import time


class Trainer:
    """训练器核心类，封装完整训练流程"""

    def __init__(self, args):
        """初始化训练器
        Args:
            args: 包含所有训练参数的命名空间对象
        """
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # 自动选择设备
        self.setup_directories()  # 创建输出目录
        self.writer = self.setup_tensorboard()  # 初始化TensorBoard
        self.model: Model = self.init_model()  # type: ignore 初始化模型
        self.criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr
        )  # Adam优化器
        # 动态学习率调度器（根据验证损失调整）
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.75, patience=3
        )
        self.train_loader, self.val_loader = (
            self.prepare_data_loaders()
        )  # 准备数据加载器

    def setup_directories(self):
        """创建模型保存目录并清理旧日志"""
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")  # 时间戳用于版本管理
        self.model_dir = f"models/{self.timestamp}"  # 模型保存路径
        os.makedirs(self.model_dir, exist_ok=True)  # 确保目录存在
        shutil.rmtree("tf-logs", ignore_errors=True)  # 清理旧TensorBoard日志

    def setup_tensorboard(self):
        """初始化TensorBoard日志记录器"""
        if self.args.use_tensorboard:
            log_dir = f"tf-logs/{self.timestamp}_{self.args.model_name}"
            return SummaryWriter(log_dir=log_dir)  # 创建SummaryWriter实例
        return None

    def init_model(self):
        """初始化模型架构"""
        model = Model(
            num_classes=2,  # 二分类任务
            freeze_backbone=self.args.freeze_backbone,  # 是否冻结主干网络
            model_name=self.args.model_name,  # 模型架构名称
        ).to(self.device)

        # 可选模型编译（PyTorch 2.0+特性）
        if self.args.compile:
            model = torch.compile(model)
        return model

    def prepare_data_loaders(self):
        """准备训练和验证数据加载器"""
        # 划分训练集和验证集（8:2比例）
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # 训练数据加载器（启用shuffle加速）
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

        # 验证数据加载器（不需要shuffle）
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )
        return train_loader, val_loader

    def save_checkpoint(self, epoch, loss, accuracy, step, is_best=False):
        """保存模型检查点
        Args:
            epoch: 当前epoch数
            loss: 当前损失值
            accuracy: 当前准确率
            is_best: 是否当前最佳模型
        """
        metadata = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "args": vars(self.args),  # 训练参数存档
            "device": str(self.device),
            "model_name": self.args.model_name,
            "timestamp": time.time(),
            "seed": torch.initial_seed(),
        }
        checkpoint = {
            "model": self.model.state_dict(),  # type: ignore 模型参数
            "optimizer": self.optimizer.state_dict(),  # 优化器状态
        }

        # 标准检查点命名（包含epoch和loss信息）
        filename = f"{self.model_dir}/Epoch{epoch:03d}_{step:04d}.pth"
        torch.save(checkpoint, filename)
        json.dump(metadata, open(f"{self.model_dir}/Epoch{epoch:03d}_{step:04d}", "w"))
        # 如果是当前最佳模型，额外保存副本
        if is_best:
            best_path = f"{self.model_dir}/best_model.pth"
            shutil.copyfile(filename, best_path)
            json.dump(metadata, open(f"{self.model_dir}/best_model.json", "w"))

    def train_epoch(self, epoch):
        """执行单个epoch的训练
        Args:
            epoch: 当前epoch序号
        Returns:
            本epoch的平均训练损失
        """
        self.model.train()
        losses = []
        # 使用tqdm创建进度条
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}"
        )

        for batch_idx, (data, target) in enumerate(progress_bar):
            # 数据转移到指定设备
            data, target = data.to(self.device), target.to(self.device)

            # 标准训练步骤
            self.optimizer.zero_grad()  # 清零梯度
            output = self.model(data)  # 前向传播
            loss = self.criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 参数更新

            # 记录损失并更新进度条
            losses.append(loss.item())
            progress_bar.set_postfix(
                {"loss": f"{np.mean(losses[-10:]):.4f}"}
            )  # 显示最近10个batch的平均损失

            # TensorBoard记录（如果启用）
            if self.writer:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("Loss/train_step", loss.item(), global_step)

            # 定期保存检查点
            if (
                epoch * len(self.train_loader) + batch_idx
            ) % self.args.checkpoint_interval == 0:
                self.save_checkpoint(epoch, loss.item(), 0, batch_idx)

        return np.mean(losses)  # 返回本epoch平均损失

    def validate(self):
        """在验证集上评估模型性能
        Returns:
            val_loss: 平均验证损失
            accuracy: 分类准确率（百分比）
        """
        self.model.eval()  # 切换到评估模式
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():  # 禁用梯度计算
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                val_loss += loss.item()  # 累计损失

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total  # 计算百分比准确率
        avg_loss = val_loss / len(self.val_loader)  # 计算平均损失
        return avg_loss, accuracy

    def train(self):
        """执行完整训练流程"""
        best_accuracy = 0  # 记录最佳准确率

        for epoch in range(self.args.epochs):
            # 训练阶段
            train_loss = self.train_epoch(epoch)

            # 验证阶段
            val_loss, val_accuracy = self.validate()

            # 调整学习率（基于验证损失）
            self.scheduler.step(val_loss)

            # 检查是否当前最佳模型
            is_best = val_accuracy > best_accuracy
            if is_best:
                best_accuracy = val_accuracy

            # 保存检查点
            self.save_checkpoint(epoch, val_loss, val_accuracy, 0, is_best)

            # 记录训练指标
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/val", val_accuracy, epoch)
                self.writer.add_scalar(
                    "LR", self.optimizer.param_groups[0]["lr"], epoch
                )

            # 打印epoch摘要
            print(
                f"Epoch {epoch+1}/{self.args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2f}%"
            )

        # 训练完成后关闭TensorBoard写入器
        if self.writer:
            self.writer.close()

        # 保存最终模型
        torch.save(self.model.state_dict(), f"{self.model_dir}/final_model.pth")
        print(f"训练完成。模型已保存至 {self.model_dir}")


def parse_args():
    """解析命令行参数"""
    parser = ArgumentParser(description="PyTorch模型训练脚本")

    # 训练参数组
    train_group = parser.add_argument_group("训练参数")
    train_group.add_argument("--epochs", type=int, default=10, help="训练总轮次")
    train_group.add_argument("--batch_size", type=int, default=32, help="批次大小")
    train_group.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    train_group.add_argument(
        "--num_workers", type=int, default=4, help="数据加载工作线程数"
    )

    # 模型参数组
    model_group = parser.add_argument_group("模型参数")
    model_group.add_argument(
        "--model_name",
        type=str,
        default="resnet152",
        choices=["resnet18", "resnet50", "resnet152", "mobilenet_v3_large"],
        help="选择模型架构",
    )
    model_group.add_argument(
        "--freeze_backbone", action="store_true", help="冻结主干网络权重"
    )
    model_group.add_argument(
        "--compile", action="store_true", help="启用torch.compile()优化"
    )

    # 日志/保存参数组
    log_group = parser.add_argument_group("日志参数")
    log_group.add_argument(
        "--use_tensorboard", action="store_true", help="启用TensorBoard记录"
    )
    log_group.add_argument(
        "--checkpoint_interval", type=int, default=100, help="检查点保存间隔（步数）"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # 清空GPU缓存
    torch.cuda.empty_cache()

    # 解析命令行参数
    args = parse_args()

    # 初始化并运行训练器
    trainer = Trainer(args)
    trainer.train()
