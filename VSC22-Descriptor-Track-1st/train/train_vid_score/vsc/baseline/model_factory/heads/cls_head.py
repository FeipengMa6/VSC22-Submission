import torch
from torch import nn

from ..utils import HEADS

from .base import BaseHead


@HEADS.register_module()
class VITCLSHead(BaseHead):

    def __init__(self,
                 feat_dim=768,
                 output_dim=512,
                 init_std=0.01,
                 **kwargs):

        super().__init__(num_classes=kwargs.get("num_classes"), in_channels=kwargs.get("in_channels"), **kwargs)

        self.feat_dim = feat_dim
        self.init_std = init_std
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        self.apply(self._init_weights)

    # @torchsnooper.snoop()
    def _compute_loss(self, feat_infos, aux_infos, local_rank, world_size):

        query_text = aux_infos['query_text']
        doc_text = aux_infos['doc_text']
        query_feat = feat_infos["query_feat"]
        doc_feat = feat_infos["doc_feat"]
        be_masked = aux_infos["be_masked"]

        query_feat, doc_feat, query_text, doc_text, be_masked = self.all_gather(
            local_rank=local_rank, world_size=world_size, query_feat=query_feat, doc_feat=doc_feat,
            query_text=query_text, doc_text=doc_text, be_masked=be_masked[:, None]
        )
        be_masked = be_masked.squeeze(1)

        logits = query_feat @ doc_feat.t() / self.temperature  # N, N

        with torch.no_grad():
            masks = query_text[:, None, :] - query_text[None, :, :]  # N, 1, L - 1, N, L = N, N, L
            masks = masks.abs().sum(dim=-1).eq(0).float()  # N, N
            diagonals = torch.eye(logits.size(0)).to(logits.device)  # N, N
        logits = logits + (masks - diagonals) * -10000

        bz = logits.size(0)
        gt = torch.arange(bz, dtype=torch.int64, device=logits.device)

        ct_loss = self.criterion(logits, gt)  # N

        length_mask = doc_text.sign().sum(dim=1).gt(2)

        mask = (length_mask * be_masked).bool()

        valid_loss = ct_loss[mask]
        loss = valid_loss.mean()

        losses = {
            "loss": loss
        }

        return losses

    def _forward_feats(self, feats, aux_infos=None, local_rank=None, world_size=None, is_train=True, is_inference=False):

        outputs = feats[:, 0]  # cls token

        return outputs

    def forward(self, feats, aux_infos=None, local_rank=None, world_size=None, is_train=True, is_inference=False):

        feat_infos = self._forward_feats(
            feats=feats, aux_infos=aux_infos, local_rank=local_rank, world_size=world_size, is_train=is_train,
            is_inference=is_inference
        )
        return feat_infos

        # if is_inference:
        #     return feat_infos
        #
        # losses = self._compute_loss(feat_infos, aux_infos, local_rank, world_size)
        #
        # return losses

