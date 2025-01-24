from dataset import transform, dataset
from model import Model
from PIL import Image
import torch
import gradio as gr

# 加载模型
model_path = "./model/res152/Model_0.1077(1027图).pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(2)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()


# 定义预测函数
def predict(img):
    try:
        img = transform(img)  # 应用预处理
        img = img.unsqueeze(0).to(device)  # 添加批量维度并移动到设备
        with torch.no_grad():
            output = model(img)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
            confidence = probabilities[0].tolist()

            # 格式化置信度为字符串
            confidence_str = ", ".join(
                [f"Class {i}: {p:.2f}" for i, p in enumerate(confidence)]
            )

            # 解析结果
            result = [predicted_class, confidence_str]
            return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"Predicted Class": "Error", "Confidence": str(e)}


# 定义 Gradio 接口
iface = gr.Interface(
    fn=predict,  # 预测函数
    inputs=gr.Image(type="pil"),  # 输入为 PIL 图像
    outputs=[
        gr.Text(label="Predicted Class"),  # 显示预测类别
        gr.Text(label="Confidence"),  # 显示置信度
    ],
    title="Image Classifier",  # 页面标题
    description="Upload an image to classify it using the trained model.",  # 页面描述
)

# 启动 Gradio 应用
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8080)
