import os
from os.path import join
import torch
from torch.nn import init
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from network.network_builder import get_gazelle_model
import yaml


def predict(image_path, model, transform, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output, 1)
    return prediction.item()


# test feeding
if __name__ == "__main__":
    RANDOM_WEIGHT = 0

    with open('configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')

    model, transforms = get_gazelle_model(config)

    if RANDOM_WEIGHT:
        for name, param in model.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean=0.0, std=0.02)  # Normal distribution
            elif 'bias' in name:
                init.constant_(param, 0.0)  # Set biases to zero (optional)
    else:
        model.load_gazelle_state_dict(torch.load(config['inference']['checkpoint'], weights_only=True),
                                      include_backbone=False)

    model.to(device)

    #transform = Compose([ToTensor(), Normalize(mean=config['data']['mean'], std=config['data']['std'])])

    mock_img = torch.randn(1, 3, 448, 448)
    mock_head_prompt = None
    mock_input = {"images": mock_img.to(device),
                  "bboxes": [[mock_head_prompt]]
                  }

    print(model)

    with torch.no_grad():
        preds = model(mock_input)

    head_id = 0

    predicted_heatmap = preds["heatmap"][0][head_id]
    predicted_inout = preds["inout"][0][head_id]

    # output Tensor of [64,64] for 1st head in 1st image of the batch; inout in range [0,1] -- 1 in-frame; 0 out-of-frame
    print(predicted_heatmap.shape, predicted_inout)

