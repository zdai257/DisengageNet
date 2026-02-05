import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from timm.models.vision_transformer import Block
import math


# Official DINOv2 backbones from torch hub (https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub)
class DinoV2Backbone(nn.Module):
    def __init__(self, model_name):
        super(DinoV2Backbone, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2)  # "b (out_h out_w) c -> b c out_h out_w"
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
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Resize(in_size),
        ])


class DinoV2BackboneMultiScale(nn.Module):
    def __init__(self, model_name, num_scales=3):
        super().__init__()
        self.base_backbone = DinoV2Backbone(model_name)
        # Store the desired number of scales
        self.num_scales = num_scales
        if self.num_scales < 1:
            raise ValueError("num_scales must be at least 1")

    def forward(self, x):
        # Obtain the original feature map [B, C, H, W]
        features = self.base_backbone.forward(x)
        multi_scale_features = []
        current_features = features
        for i in range(self.num_scales):
            if i == 0:
                # First scale is the original feature map
                multi_scale_features.append(current_features)
            else:
                # Subsequent scales are downsampled
                # Using 0.5^i as scale factor relative to the original
                scale_factor = 0.5 ** i
                downsampled_features = nn.functional.interpolate(
                    features, scale_factor=scale_factor, mode='bilinear', align_corners=False
                )
                multi_scale_features.append(downsampled_features)
        # Return a list of feature maps
        return multi_scale_features

    def get_out_size(self, in_size):
        return self.base_backbone.get_out_size(in_size)

    def get_multi_scale_channels(self):
        C = self.base_backbone.get_dimension()
        # Return a list of C repeated num_scales times
        return [C] * self.num_scales

    def get_transform(self, size):
        return self.base_backbone.get_transform(size)


def repeat_tensors(tensor, repeat_counts):
    repeated_tensors = [tensor[i:i+1].repeat(repeat, *[1] * (tensor.ndim - 1)) for i, repeat in enumerate(repeat_counts)]
    return torch.cat(repeated_tensors, dim=0)


def split_tensors(tensor, split_counts):
    indices = torch.cumsum(torch.tensor([0] + split_counts), dim=0)
    return [tensor[indices[i]:indices[i+1]] for i in range(len(split_counts))]


class MultiScaleFusionLite(nn.Module):
    def __init__(self, in_channels_list, out_channels, target_size):
        """
        Args:
            in_channels_list: List of channel dimensions for each feature map.
            out_channels: Desired number of channels after fusion.
            target_size: Tuple (height, width) for spatial alignment.
        """
        super().__init__()
        self.target_size = target_size
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        # Learnable scalar weights for each scale
        self.scale_weights = nn.Parameter(torch.ones(len(in_channels_list)))
        self.refine_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, feature_maps):
        processed_maps = []
        for conv, feat in zip(self.convs, feature_maps):
            feat_proj = conv(feat)
            feat_resized = nn.functional.interpolate(feat_proj, size=self.target_size, mode='bilinear',
                                                     align_corners=False)
            processed_maps.append(feat_resized)
        weights = torch.softmax(self.scale_weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, processed_maps))
        fused = self.refine_conv(fused)
        return fused


class MoELayer(nn.Module):
    def __init__(self, in_features, out_features, num_experts=8, num_shared_experts=2, top_k=2, hidden_dim=None):
        super().__init__()
        self.num_experts = num_experts  # Routed experts
        self.num_shared_experts = num_shared_experts  # Shared experts
        self.top_k = top_k
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_features * 4

        # Routed expert networks
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, out_features)
            ) for _ in range(num_experts)
        ])

        # Shared expert networks
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, out_features)
            ) for _ in range(num_shared_experts)
        ])

        # Gating network for routed experts only
        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        # x: [batch_size, seq_len, in_features] or [batch_size, in_features]
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_features)  # [batch_size * seq_len, in_features]

        # Initialize output
        output = torch.zeros(x_flat.shape[0], self.out_features, device=x.device)

        # Shared experts: always applied
        for expert in self.shared_experts:
            output += expert(x_flat) / (self.num_shared_experts + 1e-10)  # Average shared contributions

        # Routed experts: top-k selection
        gate_logits = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
        gate_weights = torch.softmax(gate_logits, dim=-1)  # [batch_size * seq_len, num_experts]
        top_k_weights, top_k_indices = gate_weights.topk(self.top_k, dim=-1)  # [batch_size * seq_len, top_k]
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-10)  # Normalize

        # Compute weighted sum of routed expert outputs
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # [batch_size * seq_len]
            weights = top_k_weights[:, k].unsqueeze(-1)  # [batch_size * seq_len, 1]
            for i in range(self.num_experts):
                mask = (expert_idx == i).float().unsqueeze(-1)  # [batch_size * seq_len, 1]
                expert_output = self.routed_experts[i](x_flat)  # [batch_size * seq_len, out_features]
                output += mask * weights * expert_output

        # Reshape back to original shape
        output = output.view(*batch_shape, self.out_features)
        return output


class MoEBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0.1, num_experts=8, num_shared_experts=2, top_k=2):
        super().__init__(dim, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path)
        # Replace the FFN (self.mlp) with MoELayer
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MoELayer(
            in_features=dim,
            out_features=dim,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            hidden_dim=hidden_dim
        )


# Shared Transformer (for 'vanilla' decoder block)
class SharedTransformer(nn.Module):
    def __init__(self, transformer_block, num_layers):
        super().__init__()
        self.block = transformer_block  # A single transformer block (e.g., vanilla Block)
        self.num_layers = num_layers

    def forward(self, x):
        for _ in range(self.num_layers):
            x = self.block(x)
        return x


class GazeMoE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, mlp_ratio=4, num_layers=3, in_size=(448, 448), out_size=(64, 64),
                 num_experts=8, num_shared_experts=2, top_k=2, dropout=0.1, moe_type="vanilla", is_msf=False):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        if not is_msf:
            self.ms_fusion = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        else:
            # Multi-scale fusion module (lightweight version)
            multi_scale_channels = backbone.get_multi_scale_channels()
            self.ms_fusion = MultiScaleFusionLite(
                in_channels_list=multi_scale_channels,
                out_channels=self.dim,
                target_size=(self.featmap_h, self.featmap_w)
            )

        self.register_buffer("pos_embed",
                             positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(
                                 dim=0))

        if moe_type == "vanilla":
            self.transformer = nn.Sequential(*[
                Block(dim=self.dim, num_heads=8, mlp_ratio=self.mlp_ratio, drop_path=dropout)
                for _ in range(num_layers)
            ])
        elif moe_type == "shared":
            # Create one vanilla block and share it across num_layers iterations.
            vanilla_block = Block(dim=self.dim, num_heads=8, mlp_ratio=self.mlp_ratio, drop_path=dropout)
            self.transformer = SharedTransformer(vanilla_block, num_layers)
        else:
            # Create Transformer blocks with MoE
            self.transformer = nn.Sequential(*[
                MoEBlock(
                    dim=self.dim,
                    num_heads=8,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=dropout,
                    num_experts=self.num_experts,
                    num_shared_experts=self.num_shared_experts,
                    top_k=self.top_k
                ) for _ in range(num_layers)
            ])

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
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        # Multi-scale features and fusion
        feats = self.backbone.forward(input["images"])
        x = self.ms_fusion(feats)  # [B, dim, featmap_h, featmap_w]

        x = x + self.pos_embed
        x = repeat_tensors(x, num_ppl_per_img)
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]

        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = split_tensors(x, num_ppl_per_img)

        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:
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

    def get_gazemoe_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        else:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}

    def load_gazemoe_state_dict(self, ckpt_state_dict, include_backbone=False):
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


def positionalencoding2d(d_model, height, width):
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


def gazemoe_dinov2_vitl14_inout(bbtype, d_model, mlp_ratio, num_layers, num_experts, num_shared_experts, top_k, dropout,
                                moe_type, is_msf):
    if bbtype == "DINOv2":
        backbone = DinoV2BackboneMultiScale('dinov2_vitl14', num_scales=is_msf)
    else:
        raise TypeError("backbone not supported!")

    transform = backbone.get_transform((448, 448))
    model = GazeMoE(backbone, inout=True, dim=d_model, mlp_ratio=mlp_ratio, num_layers=num_layers,
                    num_experts=num_experts,
                    num_shared_experts=num_shared_experts, top_k=top_k, dropout=dropout,
                    moe_type=moe_type, is_msf=is_msf)
    return model, transform


def get_gazemoe_model(configuration=None):
    factory = {
        "gazemoe_dinov2_vitl14_inout": gazemoe_dinov2_vitl14_inout,
    }
    return factory["gazemoe_dinov2_vitl14_inout"](
        bbtype='DINOv2',
        d_model=256,
        mlp_ratio=1,
        num_layers=3,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        dropout=0.1,
        moe_type='moe',
        is_msf=1,
    )
