from __future__ import annotations
import torch
import os


# 1. Path to the cloned dinov3 repository
REPO_DIR = '/home/CAMPUS/daiz1/repo/dinov3'

# 2. Path to the downloaded .pth checkpoint file
#WEIGHTS_PATH = 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth' 
WEIGHTS_PATH = 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'

# 3. Model entrypoint name from hubconf.py
MODEL_ENTRYPOINT = 'dinov3_vitl16' 

# ---------------------

print(f"Loading model from local repo: {REPO_DIR}")

# Load the model locally
# source='local' tells torch.hub to use the local REPO_DIR
# weights=<path> tells the model's hubconf function where to find the weights
model = torch.hub.load(
    REPO_DIR, 
    MODEL_ENTRYPOINT, 
    source='local', 
    weights=WEIGHTS_PATH
)

print(model)

# Set the model to evaluation mode
model.eval()

print("DINOv3 model loaded successfully.")


