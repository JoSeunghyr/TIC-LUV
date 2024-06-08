import torch.nn as nn
import torch
from functools import partial
from .vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, trunc_normal_
from .helpers import load_pretrained
from .build import MODEL_REGISTRY
from .dmuv import DMUV
from .tica import TICA


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 2, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@MODEL_REGISTRY.register()
class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained = True
        patch_size = 16
        self.model = DMUV(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x

@MODEL_REGISTRY.register()
class TIC_LUV(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, num_classes=2, num_frame=16, attention_type='divided_space_time', pretrained_model='', **kwargs):
        super(TIC_LUV, self).__init__()
        self.pretrained = True
        self.model = DMUV(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12,
                          num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frame=num_frame,
                          attention_type=attention_type, **kwargs)
        self.model_tic = TICA(input_size=6, num_channels=[25]*4, kernel_size=2, dropout=0.05)
        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frame, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
        # Classifier head
        self.first_head = nn.Linear(embed_dim + 25 + 4, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, x1, tic, cp):  # x0:BTCHW  x,x1:BCTHW
        x_vd, x_vd2tic = self.model(x, x1)  # x_vd: B,768  x_vd2tic: B,25
        x_tic = self.model_tic(tic)  # x_tic: B,25
        x = torch.cat((x_vd, x_tic, cp), 1)
        x = self.first_head(x)
        x = self.head(x)
        return x, x_vd2tic, x_tic
