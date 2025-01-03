import os
from os.path import join
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from network.network_builder import build_model
import yaml


def predict(image_path, model, transform, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output, 1)
    return prediction.item()


if __name__ == "__main__":
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config['model'])
    model.load_state_dict(torch.load(config['eval']['checkpoint']))
    model.to(device)

    transform = Compose([ToTensor(), Normalize(mean=config['data']['mean'], std=config['data']['std'])])

    image_path = "path/to/image.jpg"
    prediction = predict(image_path, model, transform, device)
    print(f"Predicted Label: {prediction}")

