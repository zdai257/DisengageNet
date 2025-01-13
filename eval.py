import os
from os.path import join
import torch
from utils.dataset_builder import get_dataset_and_loader
from network.network_builder import build_model
import yaml


if __name__ == "__main__":
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config['model'])
    model.load_state_dict(torch.load(config['eval']['checkpoint']))
    model.to(device).eval()

    _, test_loader = get_dataset_and_loader(config['data']['test_path'], config, train=False)

    accuracy = 0
    total = 0

    # TODO: eval best.models
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {accuracy / total * 100:.2f}%")
