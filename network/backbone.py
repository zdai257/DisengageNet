from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision.transforms as transforms


# Abstract Backbone class
class Backbone(nn.Module, ABC):
    def __init__(self):
        super(Backbone, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transform(self):
        pass


# Official DINOv2 backbones from torch hub (https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub)
class DinoV2Backbone(Backbone):
    def __init__(self, model_name):
        super(DinoV2Backbone, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2) # "b (out_h out_w) c -> b c out_h out_w"
        return x
    
    def get_dimension(self):
        return self.model.embed_dim
    
    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.model.patch_size, w // self.model.patch_size)
    
    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
            transforms.Resize(in_size),
        ])

# DINOv3 (https://github.com/facebookresearch/dinov3)
class DinoV3Backbone(Backbone):
    def __init__(self, model_name='dinov3_vitl16'):
        super(DinoV3Backbone, self).__init__()
        self.patch_size = 16
        REPO_DIR = '../dinov3/'

        if model_name == "dinov3_vitb16":
            WEIGHTS_PATH = './dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
            self.model = torch.hub.load(
            REPO_DIR,
            model_name,
            source='local',
            weights=WEIGHTS_PATH
            )
        elif model_name == "dinov3_vitl16":
            WEIGHTS_PATH = './dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
            self.model = torch.hub.load(
            REPO_DIR,
            model_name,
            source='local',
            weights=WEIGHTS_PATH
            )
        else:
            raise TypeError("DINOv3 model name undefined.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']
        
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2)
        '''
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))

        # 1. Patch Embedding
        x = self.model.patch_embed(x)
        # x shape is now (B, num_patches, embed_dim), e.g., (B, 1024, 1024)

        # 2. Reshape x to (B, H*W, C) for the transformer blocks
        x = x.flatten(2).transpose(1, 2) # x is now (60, 784, 1024)

        # 3. Generate the positional coordinates (as shown above)
        # You might need to move this generation to the device of 'x'
        H, W = x.shape[1], x.shape[2]
        H_coords = torch.arange(H, device=x.device)
        W_coords = torch.arange(W, device=x.device)
        grid_y, grid_x = torch.meshgrid(H_coords, W_coords, indexing='ij')
        coords_2d = torch.stack((grid_y, grid_x), dim=2) 
        rope_input = coords_2d.flatten(0, 1) # Shape (H*W, 2) -> (784, 2)

        rope_factors = self.rope_embed(rope_input) 

        # 3. Pass through all Transformer Blocks *except* the last one
        num_blocks = len(self.model.blocks)
        for i in range(num_blocks - 1):
            x = self.model.blocks[i](x)

        # 4. Pass through the LAST Transformer Block
        # This gives the feature output before final normalization and head.
        x = self.model.blocks[num_blocks - 1](x)
        # x shape is (B, num_patches, embed_dim)

        # 5. Final Normalization (Equivalent to 'x_norm_patchtokens' in DINOv2 if we exclude the CLS token)
        # The DINOv3 architecture shows a final `(norm)` layer, which is applied before the head.
        # This normalization is often applied *after* the last block's residual connection.
        x = self.model.norm(x) 
        # x shape is still (B, num_patches, embed_dim) -> (B, out_h * out_w, 1024)

        # 6. Reshape to mimic DINOv2 output: b c out_h out_w
        # Current shape: (B, num_patches, C) -> (B, out_h * out_w, 1024)
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2) 
        # New shape: (B, C, out_h, out_w) -> (B, 1024, out_h, out_w)
        '''
        return x

    def get_dimension(self):
        print('dimension: ', self.model.embed_dim)
        return self.model.embed_dim

    def get_out_size(self, in_size):
        h, w = in_size
        patch_size = self.patch_size
        out_h = h // patch_size
        out_w = w // patch_size
        return (out_h, out_w)

    def get_transform(self, in_size):
        return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize(in_size),])
    