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
    resnet18,
    ResNet18_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
import torch.nn as nn
import torch


class Model(nn.Module):
    model_map = {
        "resnet152": [resnet152, ResNet152_Weights.IMAGENET1K_V2],
        "resnet50": [resnet50, ResNet50_Weights.IMAGENET1K_V2],
        "resnet34": [resnet34, ResNet34_Weights.IMAGENET1K_V1],
        "resnet18": [resnet18, ResNet18_Weights.IMAGENET1K_V1],
        "mobilenet_v3_large": [
            mobilenet_v3_large,
            MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        ],
        "mobilenet_v3_small": [
            mobilenet_v3_small,
            MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        ],
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
        self.model_name = model_name

        # Load model with pretrained weights
        self.net = self.model_map[model_name][0](
            weights=self.model_map[model_name][1] if use_pretrained else None
        )

        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.net.parameters():
                param.requires_grad = False

        # Replace classifier/FC layer based on model type
        if model_name.startswith("resnet"):
            # For ResNet models
            in_features = self.net.fc.in_features
            self.net.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        elif model_name.startswith("mobilenet_v3"):
            # Keep the feature extraction part (everything before classifier)
            self.features = self.net.features

            # Get correct input features size
            in_features = self.net.classifier[0].in_features
            self.net.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        elif model_name == "efficientnet_v2_s":
            # For EfficientNet
            in_features = self.net.classifier[-1].in_features
            self.net.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        if self.model_name.startswith("mobilenet_v3"):
            # Special handling for MobileNetV3
            x = self.features(x)
            x = self.net.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.net.classifier(x)
        else:
            x = self.net(x)
        return x
