from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__()

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call."""

    @staticmethod
    def all_gather(local_rank, world_size, **tensors):

        tensors = list(tensors.values())
        _dims = [t.shape[-1] for t in tensors]
        tensors = torch.cat(tensors, dim=-1)
        with torch.no_grad():
            tensorsAll = [torch.zeros_like(tensors) for _ in range(world_size)]
            dist.all_gather(tensorsAll, tensors)
        tensorsAll[local_rank] = tensors
        tensorsAll = torch.cat(tensorsAll, dim=0)

        results = list()
        dimStart = 0
        assert sum(_dims) == tensorsAll.shape[-1]
        for D in _dims:
            results.append(tensorsAll[..., dimStart: dimStart + D])
            dimStart += D

        return tuple(results)

    def loss(self, **kwargs):
        pass
