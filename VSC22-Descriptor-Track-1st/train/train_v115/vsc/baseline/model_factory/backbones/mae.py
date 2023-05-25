from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from mmcv.runner import load_checkpoint
import torch.utils.checkpoint
import yaml
import os
from ..utils import BACKBONES


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, gradient_checkpointing=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.gradient_checkpointing = gradient_checkpointing
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for blk in self.blocks:
            # x = blk(x)

            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(blk), x)
            else:
                x = blk(x)

        x = self.norm(x)
        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x[:, 0]

        return x


@BACKBONES.register_module()
class MAE(nn.Module):

    def __init__(self,
                 pretrained='',
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 output_dim=512,
                 p=3.,
                 gradient_checkpointing=False
                 ):
        super().__init__()
        self.vit = VisionTransformer(
            patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), gradient_checkpointing=gradient_checkpointing)

        self.pretrained = pretrained
        self.output_proj = nn.Linear(embed_dim, output_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.p = p

    def init_weights(self):
        # pass
        if isinstance(self.pretrained, str):
            msg = f'load model from: {self.pretrained}'
            print(msg)
            state_dict = torch.load(self.pretrained, map_location="cpu")["model"]
            self.vit.load_state_dict(state_dict, strict=False)
            # load_checkpoint(self.vit, self.pretrained, strict=False)
        elif self.pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x_feat = self.vit.forward_features(x)
        gem = self.gem(self.fc(x_feat[:, 1:]), self.p)
        pool = self.output_proj(gem)

        return pool

    @staticmethod
    def gem(x, p=3., eps=1e-6):
        return x.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)