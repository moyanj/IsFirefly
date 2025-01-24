from dataset import transform, dataset
from model import Model
from PIL import Image
import torch

model_path = "model/res152/Model_0.1077(1027图).pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(2)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()


def predict(img):
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        return predicted.item()


if __name__ == "__main__":
    img = Image.open("dataset/no/0bb7d299.jpg")
    print(predict(img))
