import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
from mmcv.runner import load_checkpoint
from ..utils import BACKBONES


@BACKBONES.register_module()
class EfficientNet(nn.Module):

    def __init__(self,
                 arch="tf_efficientnetv2_m_in21ft1k",
                 fc_dim=256,
                 pretrained='',
                 p: float = 1.0,
                 eval_p: float = 1.0,
                 ):

        super().__init__()
        self.pretrained = pretrained
        self.backbone = timm.create_model(arch, features_only=True)
        self.fc = nn.Linear(
            self.backbone.feature_info.info[-1]["num_chs"], fc_dim, bias=False
        )
        self.bn = nn.BatchNorm1d(fc_dim)
        self.p = p
        self.eval_p = eval_p

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        # pass
        if isinstance(self.pretrained, str):
            msg = f'load model from: {self.pretrained}'
            print(msg)
            # Directly load model.
            load_checkpoint(self, self.pretrained, strict=True)
        elif self.pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.backbone(x)[-1]
        p = self.p if self.training else self.eval_p
        x = self.gem(x, p).view(batch_size, -1)
        x = self.fc(x)
        x = self.bn(x)
        # x = F.normalize(x)
        return x

    @staticmethod
    def gem(x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
