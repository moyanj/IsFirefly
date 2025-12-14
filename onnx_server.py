from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import os

app = Flask(__name__)

id_to_class = {1: "acg", 0: "不是"}
model_map = {"v1s": "model/IsACG_v1s_98.94%.onnx"}


class ONNXInference:
    def __init__(self, onnx_path, device="cpu"):
        """
        :param onnx_path: ONNX 模型路径
        :param device: 'cpu' 或 'cuda'（需要安装 onnxruntime-gpu）
        """
        self.device = device
        # 初始化 ONNX 会话
        providers = (
            ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image):
        """
        :param image: PIL.Image 或图像文件路径
        :return: 预测结果 (predicted_class, confidence)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        img = self._preprocess(image)

        # ONNX 推理
        outputs = self.session.run(None, {self.input_name: img})
        output = outputs[0]  # 假设输出是 [batch, num_classes]

        # 后处理
        predicted = np.argmax(output, axis=1)[0]  # type: ignore
        probabilities = self._softmax(output[0])  # type: ignore
        confidence = float(probabilities[predicted])

        return predicted, confidence

    def _preprocess(self, image):
        """手动实现与 torchvision 相同的预处理"""
        # 1. Resize
        image = image.resize((512, 512))  # PIL 的 resize

        # 2. PIL Image → NumPy (HWC, uint8 [0,255] → float32 [0,1])
        img_np = np.array(image, dtype=np.float32) / 255.0

        # 3. Normalize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        # 4. HWC → CHW (PIL 是 HWC，模型需要 CHW)
        img_np = np.transpose(img_np, (2, 0, 1))

        # 5. 添加 batch 维度 [C,H,W] → [1,C,H,W]
        img_np = np.expand_dims(img_np, axis=0)
        return img_np

    def _softmax(self, x):
        """手动实现 softmax"""
        exp_x = np.exp(x - np.max(x))  # 防溢出
        return exp_x / exp_x.sum()


# 初始化模型
onnx_model_path = "model/IsACG_v1s_98.94%.onnx"
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

inference = ONNXInference(onnx_model_path)


@app.route("/predict", methods=["POST"])
def predict():
    # 检查是否有文件上传
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # 检查文件是否存在
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # 读取图片文件
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 进行预测
        predicted_idx, confidence = inference.predict(image)
        predicted_class = id_to_class.get(predicted_idx, "unknown")

        # 返回结果
        return jsonify(
            {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "success": True,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
