from collections import OrderedDict

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
import torch.utils.checkpoint
import yaml
import os
from ..utils import BACKBONES


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        ret = super().forward(x)
        return ret
        # orig_type = x.dtype
        # ret = super().forward(x.type(torch.float32))
        # return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, layer_freeze: int = None,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.layer_freeze = layer_freeze if layer_freeze else layers - 1
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: torch.Tensor):

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for idx, layer_module in enumerate(self.resblocks):
            if idx < self.layer_freeze:
                with torch.no_grad():
                    x = layer_module(x)
            else:
                if self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), x)
                else:
                    x = layer_module(x)

        return x


class CLIPModel(nn.Module):

    def __init__(
            self,
            input_resolution: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            output_dim: int,
            pretrained: str = None,
            layer_freeze: int = None,
            gradient_checkpointing: bool = False
    ):

        super().__init__()
        self.pretrained = pretrained

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, layer_freeze=layer_freeze,
                                       gradient_checkpointing=gradient_checkpointing)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        # self.proj = None

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):

        if isinstance(self.pretrained, str):
            msg = f'load model from: {self.pretrained}'
            print(msg)
            # Directly load model.
            load_checkpoint(self, self.pretrained, strict=False, revise_keys=[(r'^visual\.', '')])
        elif self.pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: torch.Tensor):

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x],
            dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x


# def from_pretrained(config_dir):
#
#     config_path = os.path.join(config_dir, "config.yaml")
#     ckpt_path = os.path.join(config_dir, "pytorch_model.bin")
#     with open(config_path, 'r', encoding="utf-8") as f:
#         yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
#     yaml_cfg["pretrained"] = ckpt_path
#     model = CLIPModel(**yaml_cfg)
#     model.init_weights()
#
#     return model


@BACKBONES.register_module()
class CLIP(nn.Module):

    def __init__(self,
                 pretrained='',
                 gradient_checkpointing=False,
                 ):
        super().__init__()
        self.pretrained = pretrained
        self.gradient_checkpointing = gradient_checkpointing
        self.clip = self.from_pretrained(pretrained)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

        if pretrained:
            self.pretrained = pretrained

        pass  # TODO: Replace hard coding of loading pretrained model

    def forward(self, x):
        x_feat = self.clip(x)

        return x_feat

    def from_pretrained(self, config_dir):

        config_path = os.path.join(config_dir, "config.yaml")
        ckpt_path = os.path.join(config_dir, "pytorch_model.bin")
        with open(config_path, 'r', encoding="utf-8") as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        yaml_cfg["pretrained"] = ckpt_path
        yaml_cfg["gradient_checkpointing"] = self.gradient_checkpointing
        model = CLIPModel(**yaml_cfg)
        model.init_weights()

        return model
