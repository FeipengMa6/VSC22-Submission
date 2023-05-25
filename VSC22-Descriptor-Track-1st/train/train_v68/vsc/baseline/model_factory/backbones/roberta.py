import torch.nn as nn

from transformers import BertModel

from ..utils import BACKBONES


@BACKBONES.register_module()
class Roberta(nn.Module):

    def __init__(self,
                 pretrained='',
                 output_pool=False,
                 input_embedding=False
                 ):
        """Roberta chinese
        Args:
            pretrained (Str): pretrained model path.
            output_pool (Bool):  Return pooled output or hidden states.

        """
        super().__init__()
        self.pretrained = pretrained
        self.bert = BertModel.from_pretrained(pretrained)
        self.output_pool = output_pool
        self.input_embedding = input_embedding

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

        if pretrained:
            self.pretrained = pretrained

        pass  # TODO: Replace hard coding of loading pretrained model

    def forward(self, data, mask=None, **kwargs):
        """Defines the computation performed at every call.

        Args:
            data (torch.Tensor): The input data.
            mask (torch.Tensor):  The attention mask for input text tensor.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted
                by the backbone.
        """

        if self.input_embedding:
            x_feat = self.bert(inputs_embeds=data, attention_mask=mask)
        else:
            x_feat = self.bert(input_ids=data, attention_mask=mask)
        x_feat = x_feat.pooler_output if self.output_pool else x_feat.last_hidden_state

        return x_feat
