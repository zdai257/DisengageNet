from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision.transforms as transforms
#from transformers import BeitModel, BeitConfig
#from transformers.models.beit.image_processing_beit import BeitImageProcessor


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


class CLIPBackbone(Backbone):
    def __init__(self, model_name="ViT-B/16"):
        super().__init__()
        self.model, _ = torch.hub.load('openai/CLIP', 'load', name=model_name)
        self.vision_model = self.model.visual
        self.patch_size = 16  # Fixed for ViT-B/16

    def forward(self, x):
        # Input x: [B, C, 448, 448]
        features = self.vision_model(x)
        features = features[:, 1:, :]  # Remove CLS token

        # Reshape to spatial dimensions
        out_h, out_w = 448 // self.patch_size, 448 // self.patch_size  # 28x28 for 448x448
        features = features.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2)
        return features  # [B, 512, 28, 28] for ViT-B/16

    def get_dimension(self):
        return self.vision_model.output_dim

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.patch_size, w // self.patch_size)

    def get_transform(self, in_size):
        # Replicate CLIP's preprocessing
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP-specific
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            transforms.Resize(in_size),
        ])


"""
class BeitBackbone(Backbone):
    def __init__(self, model_name='microsoft/beit-base-patch16-224-pt22k'):
        super(BeitBackbone, self).__init__()
        self.model = BeitModel.from_pretrained(model_name)
        self.model.eval()  # Set to eval mode by default
        self.patch_size = self.model.config.patch_size
        self.image_processor = BeitImageProcessor.from_pretrained(model_name)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # BEiT expects inputs in range [0,1] and normalizes internally
        outputs = self.model(x, output_hidden_states=True)

        # We'll use the last hidden state (similar to DINOv2's patch tokens)
        # Shape: (batch_size, num_patches + 1, hidden_size)
        features = outputs.last_hidden_state

        # Remove the CLS token (first token)
        features = features[:, 1:, :]

        # Reshape to spatial dimensions
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        features = features.view(b, out_h, out_w, -1).permute(0, 3, 1, 2)  # "b (h w) c -> b c h w"

        return features

    def get_dimension(self):
        return self.model.config.hidden_size

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.patch_size, w // self.patch_size)

    def get_transform(self, in_size=(448, 448)):
        # BEiT has its own normalization parameters
        return transforms.Compose([
            transforms.ToTensor(),  # Converts to [0,1] range
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Resize(in_size),
            # Normalization will be handled by BEiT's preprocessor internally
        ])

    def preprocess(self, images):
        return self.image_processor(images, return_tensors="pt")['pixel_values']
"""
