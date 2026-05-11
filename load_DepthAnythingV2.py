import torch
import sys
sys.path.append("../Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

encoder = 'vitl'   # change to 'vits' / 'vitl' as needed
model = DepthAnythingV2(**model_configs[encoder])

model.load_state_dict(torch.load(f'../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to('cuda').eval()   # or 'cpu'

print(model)

