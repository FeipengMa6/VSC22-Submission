import torch.nn as nn
import torch
from transformers import ViTModel, ViTConfig


from ..utils import BACKBONES


@BACKBONES.register_module()
class VIT(nn.Module):

    def __init__(self,
                 feat_dim=768,
                 output_dim=256,
                 pretrained='',
                 gradient_checkpointing=False,
                 p=3.
                 ):
        """Roberta chinese
        Args:
            pretrained (Str): pretrained model path.
            output_pool (Bool):  Return pooled output or hidden states.

        """
        super().__init__()
        self.pretrained = pretrained
        config = ViTConfig.from_pretrained(pretrained)
        config.gradient_checkpointing = gradient_checkpointing
        self.vit = ViTModel.from_pretrained(pretrained, config=config)
        self.p = p
        self.output_proj = nn.Linear(feat_dim, output_dim)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

        if pretrained:
            self.pretrained = pretrained

        pass  # TODO: Replace hard coding of loading pretrained model

    def forward(self, x):

        x_feat = self.vit(x).last_hidden_state

        # cls = x_feat[:, 0]
        gem = self.gem(x_feat, self.p)
        pool = self.output_proj(gem)

        return pool

    @staticmethod
    def gem(x, p=3., eps=1e-6):
        return x.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)
