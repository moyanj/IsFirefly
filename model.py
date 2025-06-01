from torchvision.models import (
    resnet152,
    ResNet152_Weights,
    resnet50,
    ResNet50_Weights,
    resnet34,
    ResNet34_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)
import torch.nn as nn


class Model(nn.Module):
    model_map = {
        "resnet152": [resnet152, ResNet152_Weights.IMAGENET1K_V2],
        "resnet50": [resnet50, ResNet50_Weights.IMAGENET1K_V2],
        "resnet34": [resnet34, ResNet34_Weights.IMAGENET1K_V1],
        "efficientnet_v2_s": [
            efficientnet_v2_s,
            EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        ],
    }

    def __init__(
        self,
        num_classes,
        freeze_backbone=False,
        dropout=0.5,
        model_name="resnet152",
        use_pretrained=True,
    ):
        super(Model, self).__init__()
        self.resnet = self.model_map[model_name][0](
            weights=self.model_map[model_name][1] if use_pretrained else None,
            num_classes=num_classes,
        )

        # 冻结ResNet-101的卷积层
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # 替换全连接层
        self.resnet.fc = nn.Sequential(  # type: ignore
            nn.Dropout(dropout), nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x
