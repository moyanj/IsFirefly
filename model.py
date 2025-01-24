from torchvision.models import resnet152, ResNet152_Weights
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False, dropout=0.5):
        super(Model, self).__init__()
        self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)

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
