import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math
import yaml
import network.utils as utils
from network.backbone import DinoV2Backbone

#############################################
# Step 1: Wrap the Backbone for Multi-scale Features
#############################################
class DinoV2BackboneMultiScale(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_backbone = DinoV2Backbone(model_name)
        # Here, I assume that the base backbone returns a single feature map and I simulate two
        # additional scales by downsampling.
    
    def forward(self, x):
        # Obtain the original feature map
        features = self.base_backbone.forward(x)  # Expected shape: [B, C, H, W]
        # Simulate additional scales by downsampling
        f1 = features  # Highest resolution (scale 1)
        f2 = nn.functional.interpolate(features, scale_factor=0.5, mode='bilinear', align_corners=False)
        f3 = nn.functional.interpolate(features, scale_factor=0.25, mode='bilinear', align_corners=False)
        # Return a list of feature maps from different scales
        return [f1, f2, f3]
    
    def get_out_size(self, in_size):
        # Use the original backbone’s method to get the output size for the highest resolution feature.
        return self.base_backbone.get_out_size(in_size)
    
    def get_multi_scale_channels(self):
        # Assuming the base backbone outputs features with a fixed channel number.
        # For a multi-scale backbone, these could differ.
        C = self.base_backbone.get_dimension()
        return [C, C, C]  # List for f1, f2, and f3

    def get_transform(self, size):
        return self.base_backbone.get_transform(size)

#############################################
# Step 2: Multi-scale Fusion Module
#############################################
class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels, target_size):
        """
        Args:
            in_channels_list: List of channel dimensions for each feature map.
            out_channels: Desired number of channels after fusion.
            target_size: Tuple (height, width) for the spatial size to which features are aligned.
        """
        super().__init__()
        self.target_size = target_size  # e.g., (featmap_h, featmap_w)
        # Create a 1x1 conv for each scale to reduce channels to out_channels
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        # Fusion layer: Concatenate features along channel dimension and fuse them.
        self.fusion_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1)
    
    def forward(self, feature_maps):
        processed_maps = []
        for conv, feat in zip(self.convs, feature_maps):
            # Project to a common channel dimension
            feat_proj = conv(feat)
            # Resize each feature map to the target spatial dimensions
            feat_resized = nn.functional.interpolate(feat_proj, size=self.target_size, mode='bilinear', align_corners=False)
            processed_maps.append(feat_resized)
        # Concatenate along the channel dimension
        fused = torch.cat(processed_maps, dim=1)
        # Fuse concatenated features to obtain a single tensor with out_channels channels
        fused = self.fusion_conv(fused)
        return fused

#############################################
# Step 3: Modify the GazeLLE Model to Use Multi-scale Fusion
#############################################
class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        # Replace single-scale linear projection with multi-scale fusion module.
        multi_scale_channels = backbone.get_multi_scale_channels()  # e.g., [C, C, C]
        self.ms_fusion = MultiScaleFusion(
            in_channels_list=multi_scale_channels,
            out_channels=self.dim,
            target_size=(self.featmap_h, self.featmap_w)
        )
        
        # Register positional encoding.
        self.register_buffer(
            "pos_embed",
            positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)
                .squeeze(dim=0).squeeze(dim=0)
        )
        
        # Transformer blocks.
        self.transformer = nn.Sequential(*[
            Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for _ in range(num_layers)
        ])
        
        # Heatmap head to produce the final output.
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.head_token = nn.Embedding(1, self.dim)
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            self.inout_token = nn.Embedding(1, self.dim)

    def forward(self, input):
        # input["images"]: [B, 3, H, W] tensor of images
        # input["bboxes"]: list of lists of bbox tuples per image in normalized coordinates
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
        # Extract multi-scale features from the backbone.
        multi_scale_feats = self.backbone.forward(input["images"])
        # Fuse the multi-scale features into a single feature map.
        x = self.ms_fusion(multi_scale_feats)  # shape: [B, dim, featmap_h, featmap_w]
        
        # Add positional encoding.
        x = x + self.pos_embed
        # Repeat image features along people dimension per image.
        x = utils.repeat_tensors(x, num_ppl_per_img)
        
        # Process head maps and add head token embeddings.
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        
        # Flatten spatial dimensions for transformer input.
        x = x.flatten(start_dim=2).permute(0, 2, 1)  # from [B, C, H, W] to [B, H*W, C]
        
        if self.inout:
            # Prepend the in/out token.
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)
        
        x = self.transformer(x)
        
        if self.inout:
            inout_tokens = x[:, 0, :] 
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]  # Remove the in/out token
        
        # Reshape back to spatial dimensions.
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)
        
        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        # bboxes: list of list of bbox tuples per image in normalized coordinates.
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:  # No bbox provided; use an empty head map.
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps
    
    def get_gazelle_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        else:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}
        
    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        if not include_backbone:
            keys1 = set([k for k in keys1 if not k.startswith("backbone")])
            keys2 = set([k for k in keys2 if not k.startswith("backbone")])
        else:
            keys1 = set(keys1)
            keys2 = set(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]
        
        self.load_state_dict(current_state_dict, strict=False)

#############################################
# Positional Encoding (unchanged)
#############################################
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model x height x width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    d_model_half = d_model // 2
    div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model_half + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

#############################################
# Step 4: Update Model Creation Functions
#############################################

def get_gt360_model(configuration):
    factory = {
        "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
        "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
        "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
        "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
    }
    assert configuration['model']['name'] in factory.keys(), "invalid model name"
    return factory[configuration['model']['name']]()

def gazelle_dinov2_vitb14():
    # Instantiate the multi-scale backbone.
    backbone = DinoV2BackboneMultiScale('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone)
    return model, transform

def gazelle_dinov2_vitl14():
    backbone = DinoV2BackboneMultiScale('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone)
    return model, transform

def gazelle_dinov2_vitb14_inout():
    backbone = DinoV2BackboneMultiScale('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True)
    return model, transform

def gazelle_dinov2_vitl14_inout():
    backbone = DinoV2BackboneMultiScale('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True)
    return model, transform
