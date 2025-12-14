from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(model_path, quantized_model_path):
    """
    动态量化 ONNX 模型
    :param model_path: 原始模型路径
    :param quantized_model_path: 量化后模型保存路径
    """
    quantize_dynamic(
        model_path,
        quantized_model_path,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to: {quantized_model_path}")


# 使用示例
quantize_onnx_model(
    "models/release/IsACG_v1_99.06%.onnx", "models/release/IsACG_v1_99.06%_int8.onnx"
)
