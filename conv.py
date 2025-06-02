import torch
import os
import argparse


def remove_unneeded_keys(checkpoint):
    for key in list(checkpoint.keys()):
        if key not in [
            "model",
            "epoch",
            "step",
            "best_accuracy",
            "losses",
            "val_loss",
            "val_accuracy",
        ]:
            checkpoint.pop(key, None)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="检查点路径")
    parser.add_argument("-v", " --version", type=str, default="v1", help="模型版本")
    parser.add_argument("-a", " --accuracy", type=str, default="00.00", help="后缀")
    parser.add_argument("-n", " --name", type=str, default="IsACG", help="模型名称")
    parser.add_argument(
        "-p", "--out_path", type=str, default="models/release", help="保存路径"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.path)
    torch.save(
        checkpoint,
        os.path.join(
            args.out_path, f"{args.name}_v{args.version}_{args.accuracy}%_raw.pt"
        ),
    )
    checkpoint = remove_unneeded_keys(checkpoint)
    torch.save(
        checkpoint,
        os.path.join(args.out_path, f"{args.name}_v{args.version}_{args.accuracy}%.pt"),
    )


if __name__ == "__main__":
    main()
