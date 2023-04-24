import torch
import video.clip as clip
from video.model import MS
from mmcv import Config
 
if __name__ == '__main__':
    checkpoint = "../../checkpoints/clip_vit-l-14"
    img_width = 224
    model = clip.from_pretrained(checkpoint)
    model.eval()
    model.cuda()
    dummy_input = torch.randn(1, 3, img_width, img_width).cuda()
    traced_script_module = torch.jit.trace(model,example_inputs=dummy_input)
    torch.jit.save(traced_script_module,"../../checkpoints/clip.torchscript.pt")

    checkpoint = "./checkpoints/epoch_9_step_500.pth"
    cfg = "./config_vid_score.py"
    cfg = Config.fromfile(cfg)
    model = MS(cfg)
    state_dict = torch.load(checkpoint,map_location='cpu')['state_dict']
    state_dict_ = dict()
    for k,v in state_dict.items():
        if(k.startswith('module.')):
            state_dict_[k[len('module.'):]] = v
        else:
            state_dict_[k] = v
    model.load_state_dict(state_dict_)
    model.eval()
    model.cuda()
    dummy_input = torch.randn(1, 256, cfg.feat_dim).cuda()
    traced_script_module = torch.jit.trace(model,example_inputs=dummy_input)
    torch.jit.save(traced_script_module,"../../checkpoints/vsm.torchscript.pt")