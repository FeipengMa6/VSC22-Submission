import torch.nn as nn

from ..utils import BACKBONES
import torchsnooper


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


@BACKBONES.register_module()
class DeConvModel(nn.Module):

    def __init__(self, dim=2048, output_dim=3):

        super().__init__()
        self.model = nn.Sequential(
            ResBlock(dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1),
            ResBlock(dim // 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1),
            ResBlock(dim // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 4, output_dim, 4, 2, 1)
        )  #

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        self.apply(self._init_weights)

    # @torchsnooper.snoop()
    def forward(self, x):
        return self.model(x)

