import torch
import os
import argparse
import torch.onnx
from model import Model


def remove_unneeded_keys(checkpoint):
    """Remove unnecessary keys from checkpoint dictionary"""
    keys_to_remove = [
        "model",
        "epoch",
        "step",
        "best_accuracy",
        "losses",
        "val_loss",
        "val_accuracy",
        "type",
        "optimizer_state",
        "scheduler_state",
        "args",
        "is_best",
        "is_last",
    ]
    return {k: v for k, v in checkpoint.items() if k not in keys_to_remove}


def convert_to_onnx(checkpoint, path, input_shape=(1, 3, 512, 512)):
    """Convert PyTorch model to ONNX format"""
    model = Model(
        num_classes=2,
        freeze_backbone=checkpoint["args"]["freeze_backbone"],
        model_name=checkpoint["args"]["model_name"],
        use_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # Important for proper ONNX export

    # Create dummy input with specified shape
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        (dummy_input,),
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size"},  # Support dynamic batch size
            "output": {0: "batch_size"},
        },
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to release")
    parser.add_argument("path", type=str, help="Checkpoint file path")
    parser.add_argument("-v", "--version", type=str, default="1", help="Model version")
    parser.add_argument(
        "-a", "--accuracy", type=str, default="00.00", help="Accuracy suffix"
    )
    parser.add_argument(
        "-n", "--name", type=str, default="IsACG", help="Model name prefix"
    )
    parser.add_argument(
        "-p",
        "--out_path",
        type=str,
        default="models/release",
        help="Output directory path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.out_path, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(args.path)

    # Save raw checkpoint
    raw_path = os.path.join(
        args.out_path, f"{args.name}_v{args.version}_{args.accuracy}%_raw.pt"
    )
    torch.save(checkpoint, raw_path)

    # Convert to ONNX
    input_shape = (1, 3, 512, 512)
    onnx_path = os.path.join(
        args.out_path, f"{args.name}_v{args.version}_{args.accuracy}%.onnx"
    )
    convert_to_onnx(checkpoint, onnx_path, input_shape)

    model_name = checkpoint["args"]["model_name"]
    # Clean and save release checkpoint
    checkpoint = remove_unneeded_keys(checkpoint)
    checkpoint["type"] = "release"
    checkpoint["model_name"] = model_name
    release_path = os.path.join(
        args.out_path, f"{args.name}_v{args.version}_{args.accuracy}%.pt"
    )
    torch.save(checkpoint, release_path)


if __name__ == "__main__":
    main()
