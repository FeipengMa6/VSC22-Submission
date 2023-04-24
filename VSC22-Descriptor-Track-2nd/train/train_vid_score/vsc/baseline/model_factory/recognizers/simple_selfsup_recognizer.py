
import numpy as np
import torch
import torch.distributed as dist

from ..utils import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class SimpleContrastRecognizer(BaseRecognizer):

    def __init__(self,
                 backbone,
                 head=None,
                 **kwargs):

        custom_heads = {'head': head} if head else None
        super().__init__(
            backbone=backbone, cls_head=None, neck=None, train_cfg=None, test_cfg=None,
            custom_backbones=None,
            custom_heads=custom_heads
        )

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()
        if hasattr(self, "head"):
            self.head.init_weights()

    def extract_aux(self, **kwargs):
        return kwargs

    def extract_feat(self, **kwargs):
        """Extract features through a backbone.

        Args:
            data (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        return self.backbone(**kwargs)

    def forward_train(self, **kwargs):
        """Defines the computation performed at every call when training."""

        local_rank, world_size = self.get_rank_world_size()

        feats = self.extract_feat(**kwargs)
        aux_infos = self.extract_aux(**kwargs)
        if hasattr(self, "head"):
            result = self.head(feats, aux_infos, local_rank, world_size)
        else:
            result = feats

        return result

    def _do_test(self, **kwargs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""

        local_rank, world_size = self.get_rank_world_size()

        feats = self.extract_feat(**kwargs)
        aux_infos = self.extract_aux(**kwargs)
        cls_score = self.head(feats, aux_infos, local_rank, world_size, is_train=False)

        return cls_score

    def _do_inference(self, **kwargs):
        """Defines the computation performed at every call
        when forwarding logits."""

        local_rank, world_size = self.get_rank_world_size()

        feats = self.extract_feat(**kwargs)
        aux_infos = self.extract_aux(**kwargs)
        if hasattr(self, "head"):
            results = self.head(
                feats, aux_infos, local_rank, world_size,
                is_train=False, is_inference=True)
        else:
            results = feats

        return results

    @staticmethod
    def get_rank_world_size():
        if dist.is_initialized() and dist.is_available():
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            local_rank = None
            world_size = None

        return local_rank, world_size

    def forward_test(self, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        results = self._do_test(**kwargs)
        if isinstance(results, dict):
            # concatenate results of all different tasks bor broadcasting
            output = []
            for key in results:
                output.append(results[key].cpu().numpy())
            results = np.concatenate(output, axis=-1)
        else:
            results = results.cpu().numpy()

        return results

    def forward_dummy(self, **kwargs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """

        return self._do_inference(**kwargs)

    def forward_inference(self, **kwargs):
        """Used for inference logits.

        Args:

        Returns:
            Tensor: Class score.
        """

        return self._do_inference(**kwargs)

    def forward_gradcam(self, **kwargs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(**kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        losses = self(
            **data_batch
        )

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """

        losses = self(
            **data_batch
        )

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def forward(self, return_loss=True, is_inference=False, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(**kwargs)
        # 在最外层计算loss
        return self.forward_inference(**kwargs)

        # if return_loss:
        #     return self.forward_train(**kwargs)
        # if is_inference:
        #     return self.forward_inference(**kwargs)

        # return self.forward_test(**kwargs)
