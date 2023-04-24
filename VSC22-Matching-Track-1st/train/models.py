import timm
import torch
from torch import nn


class ClassifyModel(nn.Module):

    def __init__(self, model_str='mobilenetv3_small_100', num_classes=2):
        super().__init__()
        assert model_str in timm.list_models(pretrained=True)
        self.model_str = model_str
        self.num_classes = num_classes
        self.model = timm.create_model(model_str, pretrained=True, num_classes=num_classes)

    def forward(self, input):
        return self.model(input)



class HRnet(torch.nn.Module):
    def __init__(self, model_str='hrnet_w18', in_chann=270, out_chann=64, out_indices=(0, 1, 2, 3, 4)):
        super().__init__()
        model = timm.create_model(model_str, pretrained=True, features_only=True, feature_location='',
                                  out_indices=out_indices)
        model.conv1.stride = (1, 1)
        model.conv2.stride = (1, 1)
        self.model = model
        upsample_list = [
            torch.nn.Identity(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Upsample(scale_factor=4, mode='nearest'),
            torch.nn.Upsample(scale_factor=8, mode='nearest')
        ]
        if 0 in out_indices:
            in_chann += 64
            upsample_list = [torch.nn.Identity()] + upsample_list
        self.upsample_list = torch.nn.ModuleList(upsample_list)
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(in_chann, out_chann, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_chann, 2, 1, 1)
        )
    def forward(self, x):
        yl = self.model(x)
        yl = [up(x) for x, up in zip(yl, self.upsample_list)]
        y = torch.cat(yl, dim=1)
        y = self.fuse(y)
        return y
