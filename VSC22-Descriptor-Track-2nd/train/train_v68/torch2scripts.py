import argparse
import torch
from vsc.baseline.model_factory.utils import build_model
from vsc.baseline.model_factory.backbones.sscd import SSCDModel
from mmcv import Config
import os


CHECKPOINT_PATH = "./checkpoints/epoch_39.pth"
IMG_WIDTH = 384
MODEL_NAME = "vit_base_patch32_384"
OUTPUT_NAME = "vit_v68"
 
if __name__ == '__main__':
    model = SSCDModel(name=MODEL_NAME,pool_param=3.,pool="gem",pretrained=None,dims=(768, 512),use_classify=False,add_head=True)
    state_dict = torch.load(CHECKPOINT_PATH,map_location='cpu')['state_dict']
    state_dict_ = dict()
    for k,v in state_dict.items():
        if(k.startswith('module.backbone.')):
            state_dict_[k[len('module.backbone.'):]] = v
        else:
            state_dict_[k] = v
    model.load_state_dict(state_dict_)
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_WIDTH, IMG_WIDTH)
    # out1 = model(dummy_input).detach().numpy()
    # print(out1.shape)
    traced_script_module = torch.jit.trace(model,example_inputs=dummy_input)
    torch.jit.save(traced_script_module,f"../../checkpoints/{OUTPUT_NAME}.torchscript.pt")
    # model = torch.jit.load(f"./{OUTPUT_NAME}.torchscript.pt")
    # model = model.eval()
    # out2 = model(dummy_input).detach().numpy()
    # print(((out1 - out2) < 1e-4).mean()) 
