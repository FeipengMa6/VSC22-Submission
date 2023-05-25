from ..utils import BACKBONES
from torch import nn
import timm
import torch
from torch.nn import functional as F
from classy_vision.models import build_model as cv_build_model
from mmcv.runner import load_checkpoint
from timm.models.vision_transformer import _load_weights


class GlobalGeMPool2d(nn.Module):
    """Generalized mean pooling.
    Inputs should be non-negative.
    """

    def __init__(
        self,
        pooling_param: float,
        linear_param = None
    ):
        """
        Args:
            pooling_param: the GeM pooling parameter
        """
        super().__init__()
        self.pooling_param = pooling_param
        self.linear_param = linear_param
        if linear_param is not None:
            self.conv = nn.Conv1d(linear_param[0], 2048, 1)
    def forward(self, x):
        if len(x.size()) == 4:
            N, C, H, W = x.size()
            x = x.reshape(N, C, H * W)  # Combine spatial dimensions
        else:
            x = x.transpose(1, 2)
        if self.linear_param is not None:
            x = self.conv(x)
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
    def __init__(self, name: str, pool_param: float = 3.0, pool="gem", use_classify=True, dims=(2048, 512), add_head=False):
        super().__init__()
        if use_classify:
            name2dims = dict(
                resnet50=(2048, 512),
                resnext101_32x4d=(2048, 512)  # 1024
            )
            dims = name2dims[name]
            self.backbone = cv_build_model({"name": name}).classy_model
        else:
            assert name in timm.list_models(pretrained=True)
            print(f'initialize with {name}')
            if 'vit_' not in name and 'xcit' not in name:
                # model = timm.create_model(name, pretrained=True, features_only=True, out_indices=(4,))
                model = timm.create_model(name, pretrained=True)
                model.global_pool = torch.nn.Identity()
                model.classifier = torch.nn.Identity()
            else:
                model = timm.create_model(name, pretrained=False, global_pool='', num_classes=0)
            # if 'grad_checkpointing' in model.__dict__:
            #     model.set_grad_checkpointing(True)
            self.backbone = model
        if add_head:
            print(f'add head: {dims}')
            embedding_seq = [
                GlobalGeMPool2d(pool_param, dims) if pool == "gem" else AvgPool2d(),
                nn.Linear(2048, dims[1])
            ]
        else:
            embedding_seq = [
                GlobalGeMPool2d(pool_param) if pool == "gem" else AvgPool2d(),
                nn.Linear(dims[0], dims[1]),
            ]
            
        self.embeddings = nn.Sequential(*embedding_seq
        )

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, list):
            x = x[-1]
        # print(x.shape)
        return self.embeddings(x)




@BACKBONES.register_module()
class SSCDModel(nn.Module):

    def __init__(self,
                 name,
                 pool_param=3.,
                 pretrained='',
                 pool="gem",
                 use_classify=True,
                 add_head=False, 
                 dims=(2048, 512)
                 ):
        super().__init__()
        self.use_classify = use_classify
        self.pretrained = pretrained
        self.model = Model(name, pool_param, pool=pool, use_classify=use_classify, dims=dims, add_head=add_head)

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
            if self.pretrained:
                msg = f'load model from: {self.pretrained}'
                print(msg)
                # Directly load model.
                # load_checkpoint(self.model, self.pretrained, strict=False)
                _load_weights(self.model.backbone,self.pretrained)
            if not self.pretrained and self.use_classify:
                raise NotImplementedError(f'pretrain: {self.pretrained}, use_classify: {self.use_classify}, not implemented')
        elif self.pretrained is None:
            print('use random init')
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        return self.model(x)