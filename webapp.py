from dataclasses import dataclass
import time
from dataset import predict_transform, dataset
from model import Model
from PIL import Image
import torch
import gradio as gr
import os

model_dir = "models/release"


@dataclass
class Result:
    class_name: int
    confidence: float


class Predictor:
    def __init__(self, model_path):
        self.model_name = ""
        self.model_path = model_path
        self.model: Model = None  # type: ignore
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.id_to_class = {1: "是", 0: "不是"}

    def load_model(self, model_name):
        model_path = os.path.join(self.model_path, model_name + ".pt")
        if not os.path.exists(model_path):
            raise ValueError(f"Model '{model_name}' not found in model_map.")
        self.model_name = model_name

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = Model(2, model_name=checkpoint["model_name"], use_pretrained=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        img = predict_transform(image)  # 应用预处理
        img = img.unsqueeze(0).to(self.device)  # 添加批量维度并移动到设备
        with torch.no_grad():
            output = self.model(img)
            _, predicted = torch.max(output, 1)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence = probabilities[0].tolist()
            return Result(
                predicted.item(),  # type: ignore
                confidence[predicted.item()],  # type: ignore
            )

    def gradio_predict(self, image, model_name):
        if model_name != self.model_name:
            self.load_model(model_name)

        # try:
        st = time.time()
        result = self.predict(image)
        print(f"Prediction time: {time.time() - st:.4f}s")
        confidence_str = f"{result.confidence * 100:.4f}%"  # type: ignore

        # 解析结果
        return [self.id_to_class[result.class_name], confidence_str]
        # except Exception as e:
        # return "Error: " + str(e), 0

    def create_gradio_interface(self):
        # 定义 Gradio 接口
        iface = gr.Interface(
            fn=self.gradio_predict,  # 预测函数
            inputs=[
                gr.Image(type="pil"),  # 输入为 PIL 图像
                gr.Dropdown(
                    choices=[i.replace(".pt", "") for i in os.listdir(model_dir)],
                    value=self.model_name,
                    label="选择模型",
                    interactive=True,
                    allow_custom_value=True,
                ),
            ],
            outputs=[
                gr.Text(label="预测类别"),  # 显示预测类别
                gr.Text(label="置信度"),  # 显示置信度
            ],
            title="是萤宝吗？",  # 页面标题
            description="上传一张图片，看看是不是萤宝",  # 页面描述
            submit_btn="检测",  # 提交按钮文本
            stop_btn="停止",
            clear_btn="清除",
        )
        return iface


# 主程序
if __name__ == "__main__":
    # 初始化一个模型实例
    predictor = Predictor(model_path="models/release")
    # 创建 Gradio 接口
    iface = predictor.create_gradio_interface()
    # 启动 Gradio 应用
    iface.launch(server_name="0.0.0.0", server_port=8080)
