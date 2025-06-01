import sys
import torch
import os

model_path = sys.argv[1]

model = torch.load(model_path)
base_name = os.path.basename(model_path)
print(base_name)
print("检查点类型：", model["type"])
print("保存时间：", model["time"])
print(f"第{model['epoch']}轮训练")
if model["type"] == "step":
    print("全局步数：", model["step"])
    print("Loss：", model["loss"])
else:
    print("测试Loss：", model["val_loss"])
    print("测试准确率：", model["val_accuracy"], "%")
    print("最好的测试准确率：", model["best_accuracy"], "%")
    print("是否为最后一个：", model["is_last"])
    print("全局步数：", model["step"])

"""
for key, value in model["args"].items():
    print(f"{key}: {value}")
"""
