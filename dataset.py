from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch

transform = transforms.Compose(
    [
        # transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = ImageFolder("dataset", transform=transform)
