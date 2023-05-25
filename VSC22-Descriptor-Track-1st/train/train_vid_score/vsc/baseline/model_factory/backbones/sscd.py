from ..utils import BACKBONES
from torch import nn
import torch
from torch.nn import functional as F
from classy_vision.models import build_model as cv_build_model
from mmcv.runner import load_checkpoint


class GlobalGeMPool2d(nn.Module):
    """Generalized mean pooling.
    Inputs should be non-negative.
    """

    def __init__(
        self,
        pooling_param: float,
    ):
        """
        Args:
            pooling_param: the GeM pooling parameter
        """
        super().__init__()
        self.pooling_param = pooling_param

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, C, H * W)  # Combine spatial dimensions
        mean = x.clamp(min=1e-6).pow(self.pooling_param).mean(dim=2)
        r = 1.0 / self.pooling_param
        return mean.pow(r)


class AvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, C, H * W)  # Combine spatial dimensions
        mean = x.clamp(min=1e-6).mean(dim=2)
        return mean


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)


class Model(nn.Module):
    def __init__(self, name: str, pool_param: float = 3.0, pool="gem"):
        super().__init__()
        name2dims = dict(
            resnet50=(2048, 512),
            resnext101_32x4d=(2048, 512)  # 1024
        )
        dims = name2dims[name]

        self.backbone = cv_build_model({"name": name}).classy_model

        self.embeddings = nn.Sequential(
            GlobalGeMPool2d(pool_param) if pool == "gem" else AvgPool2d(),
            nn.Linear(dims[0], dims[1]),
            # L2Norm(),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.embeddings(x)


@BACKBONES.register_module()
class SSCDModel(nn.Module):

    def __init__(self,
                 name,
                 pool_param=3.,
                 pretrained='',
                 pool="gem"
                 ):
        super().__init__()
        self.pretrained = pretrained
        self.model = Model(name, pool_param, pool=pool)

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
            load_checkpoint(self.model, self.pretrained, strict=False)
        elif self.pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        return self.model(x)
