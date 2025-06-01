from dataset import predict_transform, dataset
from model import Model
from PIL import Image
import torch
import gradio as gr
import os

model_dir = "models/2025-06-01_17:56"


# 定义一个类来封装模型和预测逻辑
class FireflyPredictor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id_to_class = {1: "是", 0: "不是"}
        self.load_model(model_name)

    def load_model(self, model_name):
        if not os.path.exists(os.path.join(model_dir, model_name + ".pt")):
            raise ValueError(f"Model '{model_name}' not found in model_map.")
        self.model_name = model_name
        model_path = os.path.join(model_dir, model_name + ".pt")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = Model(
            2, model_name=checkpoint["args"]["model_name"], use_pretrained=False
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model '{model_name}' loaded successfully.")

    def predict(self, img, model_name):
        if model_name != self.model_name:
            self.load_model(model_name)
        try:
            img = predict_transform(img)  # 应用预处理
            img = img.unsqueeze(0).to(self.device)  # 添加批量维度并移动到设备
            with torch.no_grad():
                output = self.model(img)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()
                confidence = probabilities[0].tolist()

                # 格式化置信度为字符串
                confidence_str = f"{confidence[predicted_class] * 100:.4f}%"  # type: ignore

                # 解析结果
                result = [self.id_to_class[int(predicted_class)], confidence_str]
                return result
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"Predicted Class": "Error", "Confidence": str(e)}

    def switch_model(self, model_name):
        if model_name != self.model_name:
            self.load_model(model_name)
            self.model_name = model_name
            print(f"Switched to model '{model_name}'")

    def create_gradio_interface(self):
        # 定义 Gradio 接口
        iface = gr.Interface(
            fn=self.predict,  # 预测函数
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
    predictor = FireflyPredictor(model_name="best_ckpt_ep001")
    # 创建 Gradio 接口
    iface = predictor.create_gradio_interface()
    # 启动 Gradio 应用
    iface.launch(server_name="0.0.0.0", server_port=8080, pwa=True)
