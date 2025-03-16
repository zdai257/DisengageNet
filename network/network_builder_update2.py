import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.vision_transformer import Block
import math
import yaml
import network.utils as utils
from network.backbone import DinoV2Backbone


class MixtureOfExperts(nn.Module):
    def __init__(self, num_routed_experts, num_shared_experts, in_channels, out_channels):
        super(MixtureOfExperts, self).__init__()
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts

        # Routed Experts
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_routed_experts)
        ])

        # Shared Experts
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_shared_experts)
        ])

        # Gating mechanism for routed experts
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, num_routed_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Compute shared expert outputs
        shared_outputs = [expert(x) for expert in self.shared_experts]
        shared_output = torch.stack(shared_outputs, dim=1).mean(dim=1)  # Average shared expert outputs

        # Compute routed expert outputs
        routed_outputs = [expert(x) for expert in self.routed_experts]
        routed_outputs = torch.stack(routed_outputs, dim=1)  # Shape: [batch, num_routed_experts, out_channels, H, W]

        # Compute gating weights for routed experts
        gate_weights = self.gate(x)  # Shape: [batch, num_routed_experts, H, W]
        gate_weights = gate_weights.unsqueeze(2)  # Shape: [batch, num_routed_experts, 1, H, W]

        # Combine routed expert outputs using gating weights
        routed_output = (routed_outputs * gate_weights).sum(dim=1)  # Shape: [batch, out_channels, H, W]

        # Combine shared and routed outputs
        output = shared_output + routed_output  # Add or concatenate, depending on the use case
        return output

#############################################
# Transformer Block Alternatives
#############################################
# 1. Linformer Block
class LinformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1, seq_len=196, k=64):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # Projection matrices to reduce sequence length for keys and values
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, L, C]
        residual = x
        x = self.norm1(x)
        B, L, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape for multi-head attention: [B, num_heads, L, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # Project keys and values to lower-dimensional sequence (k dimension)
        k_proj = torch.einsum("bhlc,lk->bhkc", k, self.E)  # [B, num_heads, k, head_dim]
        v_proj = torch.einsum("bhlc,lk->bhkc", v, self.F)  # [B, num_heads, k, head_dim]
        # Scale queries
        q = q * self.scale
        # Compute attention: [B, num_heads, L, k]
        attn = torch.einsum("bhld,bhkd->bhlk", q, k_proj)
        attn = torch.softmax(attn, dim=-1)
        # Aggregate values: [B, num_heads, L, head_dim]
        out = torch.einsum("bhlk,bhkd->bhld", attn, v_proj)
        # Reshape back to [B, L, C]
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out_proj(out)
        x = residual + out
        x = x + self.mlp(self.norm2(x))
        return x

# 2. Performer Block (Simplified Approximation)
class PerformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, L, C]
        residual = x
        x = self.norm1(x)
        B, L, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape: [B, num_heads, L, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # Apply a simple non-linearity as a feature map (here using ReLU)
        q = torch.relu(q) + 1e-6
        k = torch.relu(k) + 1e-6
        # Compute a rough linear attention: here I sum k along sequence dimension
        k_sum = k.sum(dim=2)  # [B, num_heads, head_dim]
        # Multiply q by the aggregated key information (broadcast over sequence length)
        out = q * k_sum.unsqueeze(2)  # [B, num_heads, L, head_dim]
        # Reshape back to [B, L, C]
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out_proj(out)
        x = residual + out
        x = x + self.mlp(self.norm2(x))
        return x

# 3. Shared Transformer (Parameter Sharing Across Layers)
class SharedTransformer(nn.Module):
    def __init__(self, transformer_block, num_layers):
        super().__init__()
        self.block = transformer_block  # A single transformer block (e.g., vanilla Block)
        self.num_layers = num_layers

    def forward(self, x):
        for _ in range(self.num_layers):
            x = self.block(x)
        return x

#############################################
# Step 1: Wrap the Backbone for Multi-scale Features
#############################################
class DinoV2BackboneMultiScale(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_backbone = DinoV2Backbone(model_name)
        # Here I assume the base backbone returns a single feature map,
        # and I simulate two additional scales by downsampling.
    
    def forward(self, x):
        # Obtain the original feature map [B, C, H, W]
        features = self.base_backbone.forward(x)
        # Simulate additional scales by downsampling
        f1 = features  # Highest resolution (scale 1)
        f2 = nn.functional.interpolate(features, scale_factor=0.5, mode='bilinear', align_corners=False)
        f3 = nn.functional.interpolate(features, scale_factor=0.25, mode='bilinear', align_corners=False)
        return [f1, f2, f3]
    
    def get_out_size(self, in_size):
        return self.base_backbone.get_out_size(in_size)
    
    def get_multi_scale_channels(self):
        C = self.base_backbone.get_dimension()
        return [C, C, C]
    
    def get_transform(self, size):
        return self.base_backbone.get_transform(size)

#############################################
# Step 2: Lightweight Multi-scale Fusion Module
#############################################
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
            feat_resized = nn.functional.interpolate(feat_proj, size=self.target_size, mode='bilinear', align_corners=False)
            processed_maps.append(feat_resized)
        weights = torch.softmax(self.scale_weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, processed_maps))
        fused = self.refine_conv(fused)
        return fused

#############################################
# Step 3: Modify the GazeLLE Model to Use Transformer Alternatives
#############################################
class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64),
                 transformer_type="vanilla"):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        # Multi-scale fusion module (lightweight version)
        multi_scale_channels = backbone.get_multi_scale_channels()
        self.ms_fusion = MultiScaleFusionLite(
            in_channels_list=multi_scale_channels,
            out_channels=self.dim,
            target_size=(self.featmap_h, self.featmap_w)
        )
        
        # Positional encoding
        self.register_buffer(
            "pos_embed",
            positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)
                .squeeze(dim=0).squeeze(dim=0)
        )
        
        # Choose transformer architecture based on transformer_type
        if transformer_type == "vanilla":
            self.transformer = nn.Sequential(*[
                Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
                for _ in range(num_layers)
            ])
        elif transformer_type == "linformer":
            # Use our LinformerBlock with sequence length = featmap_h * featmap_w
            seq_len = self.featmap_h * self.featmap_w
            self.transformer = nn.Sequential(*[
                LinformerBlock(dim=self.dim, num_heads=8, mlp_ratio=4, dropout=0.1, seq_len=seq_len, k=64)
                for _ in range(num_layers)
            ])
        elif transformer_type == "performer":
            self.transformer = nn.Sequential(*[
                PerformerBlock(dim=self.dim, num_heads=8, mlp_ratio=4, dropout=0.1)
                for _ in range(num_layers)
            ])
        elif transformer_type == "shared":
            # Create one vanilla block and share it across num_layers iterations.
            vanilla_block = Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            self.transformer = SharedTransformer(vanilla_block, num_layers)
        else:
            raise ValueError("Unknown transformer type: {}".format(transformer_type))
        
        # Heatmap head.
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
        # input["images"]: [B, 3, H, W]
        # input["bboxes"]: list of list of bbox tuples per image in normalized coordinates
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
        # Multi-scale features and fusion
        multi_scale_feats = self.backbone.forward(input["images"])
        x = self.ms_fusion(multi_scale_feats)  # [B, dim, featmap_h, featmap_w]
        
        x = x + self.pos_embed
        x = utils.repeat_tensors(x, num_ppl_per_img)
        
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        
        # Flatten spatial dimensions for transformer input.
        x = x.flatten(start_dim=2).permute(0, 2, 1)  # [B, H*W, dim]
        
        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)
        
        x = self.transformer(x)
        
        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]
        
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)
        
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
# TODO: GTMoE model with ConvMoE
#############################################
class GTMoE(nn.Module):
    def __init__(self, backbone, in_channels=768, num_routed_experts=4, num_shared_experts=2,
                 inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64), transformer_type=None):
        super(GTMoE).__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        # Multi-scale fusion module (lightweight version)
        multi_scale_channels = backbone.get_multi_scale_channels()
        self.ms_fusion = MultiScaleFusionLite(
            in_channels_list=multi_scale_channels,
            out_channels=self.dim,
            target_size=(self.featmap_h, self.featmap_w)
        )

        # Positional encoding
        self.register_buffer(
            "pos_embed",
            positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)
                .squeeze(dim=0).squeeze(dim=0)
        )

        if transformer_type is None:
            # Create one vanilla block and share it across num_layers iterations.
            vanilla_block = Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            self.transformer = SharedTransformer(vanilla_block, num_layers)
        else:
            raise ValueError("Unknown transformer type: {}".format(transformer_type))

        # MoEs
        self.moe1 = MixtureOfExperts(num_routed_experts, num_shared_experts, in_channels, 256)
        self.moe2 = MixtureOfExperts(num_routed_experts, num_shared_experts, 256, 128)
        self.moe3 = MixtureOfExperts(num_routed_experts, num_shared_experts, 128, 256)
        self.final_conv = nn.Conv2d(self.dim, 1, kernel_size=1)  # Output a single channel heatmap

        # Heatmap head.
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
        # input["images"]: [B, 3, H, W]
        # input["bboxes"]: list of list of bbox tuples per image in normalized coordinates
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]

        # Multi-scale features and fusion
        multi_scale_feats = self.backbone.forward(input["images"])
        x = self.ms_fusion(multi_scale_feats)  # [B, dim, featmap_h, featmap_w]

        x = x + self.pos_embed
        x = utils.repeat_tensors(x, num_ppl_per_img)

        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings

        # Flatten spatial dimensions for transformer input.
        x = x.flatten(start_dim=2).permute(0, 2, 1)  # [B, H*W, dim]

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]

        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        # MoEs
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.moe1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.moe2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.moe3(x)
        x = self.final_conv(x)

        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

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
# TODO: GTMoE model with Feed-forward MoE
#############################################
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class MoEBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, top_k):
        super(MoEBlock, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # Gating mechanism
        gate_scores = F.softmax(self.gate(x), dim=-1)
        top_k_gates, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        # Apply experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            output += top_k_gates[:, :, i].unsqueeze(-1) * expert_outputs.gather(1, top_k_indices[:, :, i].unsqueeze(
                -1).expand(-1, -1, x.size(-1)))

        return output

class GTMoE2(nn.Module):
    def __init__(self, backbone, input_dim=768, hidden_dim=256, num_routed_experts=4, num_shared_experts=2, top_k=3,
                 inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64), transformer_type=None):
        super(GTMoE2).__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        self.moe_block = MoEBlock(input_dim, hidden_dim, num_routed_experts, top_k)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.moe_block(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # Adjust dimensions for Conv2d

        pass

#############################################
# Positional Encoding (unchanged)
#############################################
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

#############################################
# Step 4: Update Model Creation Functions
#############################################

def get_gtmoe_model(configuration):
    factory = {
        "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
        "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
        "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
        "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
    }
    assert configuration['model']['name'] in factory.keys(), "invalid model name"
    return factory[configuration['model']['name']]()

def get_gt360_model(configuration):
    factory = {
        "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
        "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
        "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
        "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
    }
    assert configuration['model']['name'] in factory.keys(), "invalid model name"
    return factory[configuration['model']['name']]()

def gazelle_dinov2_vitb14(transformer_type="shared"):
    backbone = DinoV2BackboneMultiScale('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, transformer_type=transformer_type)
    return model, transform

def gazelle_dinov2_vitl14(transformer_type="shared"):
    backbone = DinoV2BackboneMultiScale('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, transformer_type=transformer_type)
    return model, transform

def gazelle_dinov2_vitb14_inout(transformer_type="shared"):
    backbone = DinoV2BackboneMultiScale('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True, transformer_type=transformer_type)
    return model, transform

def gazelle_dinov2_vitl14_inout(transformer_type="shared"):  # performer
    backbone = DinoV2BackboneMultiScale('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True, transformer_type=transformer_type)
    return model, transform

def moe_dinov2_vitl14_inout():  # performer
    backbone = DinoV2BackboneMultiScale('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GTMoE(backbone, inout=True)
    return model, transform
