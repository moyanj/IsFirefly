from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from model import Model
from dataset import dataset
from tqdm import tqdm
import os
import shutil

# 解析命令行参数
parser = ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for training"
)
parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
parser.add_argument(
    "--num_workers", type=int, default=12, help="Number of workers for DataLoader"
)
parser.add_argument(
    "--use_tensorboard", action="store_true", help="Enable TensorBoard logging"
)
args = parser.parse_args()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 TensorBoard 的 SummaryWriter（如果启用）
writer = SummaryWriter(log_dir="./tf-logs") if args.use_tensorboard else None

# 删除已存在的输出目录（如果存在）
shutil.rmtree("models", ignore_errors=True)
os.makedirs("models")


# 训练函数
def train(model, criterion, optimizer, scheduler, train_loader, val_loader):
    model.train()
    print("Training...")
    global_step = 0  # 全局步骤计数器

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False
        )
        for batch_idx, (data, label) in enumerate(progress_bar):
            data, target = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 更新进度条信息
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # 如果启用 TensorBoard，则记录每个 step 的信息
            if writer:
                writer.add_scalar("StepLoss/train", loss.item(), global_step)

            global_step += 1  # 更新全局步骤计数器

        # 计算平均损失并打印
        avg_loss = running_loss / len(train_loader)
        if writer:
            writer.add_scalar("Loss/train", avg_loss, epoch + 1)
            writer.add_scalar(
                "Learning Rate/train", optimizer.param_groups[0]["lr"], epoch + 1
            )
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] completed. Average Loss: {avg_loss:.6f}"
        )

        # 动态调整学习率
        scheduler.step(avg_loss)

        # 每个 epoch 保存一次模型
        torch.save(
            model.state_dict(), f"models/model_epoch_{epoch + 1}_{avg_loss:.4f}.pth"
        )

        # 验证逻辑
        val_loss, val_accuracy = validate(model, criterion, val_loader)
        print(
            f"Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.2f}%"
        )
        if writer:
            writer.add_scalar("Loss/validation", val_loss, epoch + 1)
            writer.add_scalar("Accuracy/validation", val_accuracy, epoch + 1)

    # 如果启用了 TensorBoard，关闭 SummaryWriter
    if writer:
        writer.close()


# 验证函数
def validate(model, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy


# 主程序
if __name__ == "__main__":
    torch.cuda.empty_cache()  # 释放显存
    # 初始化模型、损失函数、优化器和学习率调度器
    model = Model(num_classes=2, freeze_backbone=True).to(device)
    """
    model.load_state_dict(
        torch.load("model_final.pth", map_location=device, weights_only=True)
    )
    """
    print("Compiling...")
    torch.compile()
    print("Compiled.")
    print(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.75, patience=3
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 开始训练
    train(model, criterion, optimizer, scheduler, train_loader, val_loader)

    # 训练完成后保存最终模型
    torch.save(model.state_dict(), "model_final.pth")
    print("Training completed. Final model saved as 'model_final.pth'.")
