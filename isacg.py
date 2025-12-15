import argparse
import time
import onnxruntime as ort
import numpy as np
from model import Model
from PIL import Image
import torch
from dataset import predict_transform
from abc import ABC, abstractmethod


class Predictor(ABC):
    def __init__(self, model: Model, device="cpu"):
        self.model = model
        self.device = device

    @abstractmethod
    def predict(self, image: Image.Image) -> tuple[int, float]:
        """
        :param image: PIL.Image 或图像文件路径
        :return: 预测结果 (predicted_class, confidence)
        """
        pass


class ONNXInference(Predictor):
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

    def _preprocess(self, image: Image.Image):
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


class TorchInference(Predictor):
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        加载 PyTorch 模型
        :param model_path: 模型路径
        :return: 模型实例
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model = Model(2, model_name=checkpoint["model_name"], use_pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image: Image.Image) -> tuple[int, float]:
        img = predict_transform(image)  # 应用预处理
        img = img.unsqueeze(0).to(self.device)  # type: ignore 添加批量维度并移动到设备
        with torch.no_grad():
            output = self.model(img)
            _, predicted = torch.max(output, 1)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence = probabilities[0].tolist()
            return predicted.item(), confidence[predicted.item()]  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_path.endswith(".onnx"):
        predictor = ONNXInference(args.model_path, device=args.device)
    else:
        predictor = TorchInference(args.model_path, device=args.device)
    image = Image.open(args.image_path).convert("RGB")
    st = time.time()
    predicted_class, confidence = predictor.predict(image)
    print(f"Prediction time: {(time.time() - st) *1000:.4f}ms")
    print(f"Predicted class: {predicted_class}, Confidence: {confidence * 100:.4f}%")


if __name__ == "__main__":
    main()
