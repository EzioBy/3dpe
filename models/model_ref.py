import sys
sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn as nn
from segmentation_models_pytorch.decoders.deeplabv3 import DeepLabV3
import math
from einops import repeat

from functools import reduce
from segmentation_models_pytorch.encoders.mix_transformer import OverlapPatchEmbed
from models.model import EncoderEhigh, EncoderF, EncoderFinal
from models.mae_encoder import mae_encoder_vit_base_patch16


# EncoderRef to fuse features of ref-image and high-frequency features of input image by cross-attention.
class EncoderRef(nn.Module):
    def __init__(self, emb_dim=768, residual=False):
        super(EncoderRef, self).__init__()
        self.residual = residual
        self.emb_dim = emb_dim
        self.use_ref_pair = False
        self.mae_encoder = mae_encoder_vit_base_patch16()

        self.pre_conv = nn.Conv2d(in_channels=128, out_channels=96,
                                     stride=1, kernel_size=3, padding=1)
        self.patch_emb = OverlapPatchEmbed(img_size=128, in_chans=96, embed_dim=emb_dim,
                                           patch_size=7, stride=4)
        self.proj_tail = nn.Conv2d(in_channels=192, out_channels=512, stride=1, kernel_size=1, padding=0, bias=False)
        self.upsampler_ps = nn.PixelShuffle(upscale_factor=2)
        self.upsampler_bi = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x, img_ref):
        N = x.shape[0]
        x_down = self.pre_conv(x)
        x_token, H, W = self.patch_emb(x_down)
        x_ref = self.mae_encoder(img_ref)
        A = torch.matmul(x_token, x_ref.transpose(1, 2)) / math.sqrt(self.emb_dim)
        A = torch.softmax(A, dim=-1)
        attn_out = torch.matmul(A, x_ref)
        attn_out = attn_out.permute(0, 2, 1).contiguous()
        out = attn_out.view(N, self.emb_dim, H, W)
        out = self.upsampler_ps(out)
        out = self.proj_tail(out)
        out = self.upsampler_ps(out)
        if self.residual:
            out = out + x
        return out


class Live3DEditingEncoder(nn.Module):
    def __init__(self,
                 in_channels=5,
                 mode='LT',
                 encoder_name="resnet34",
                 encoder_depth=3,
                 use_bn=False,
                 guidance_type='image',
                 ref_fuse_mode='high'):
        super().__init__()
        self.mode = mode
        self.guidance_type = guidance_type
        self.ref_fuse_mode = ref_fuse_mode

        deeplab = DeepLabV3(in_channels=in_channels, encoder_name=encoder_name, encoder_depth=encoder_depth)
        self.encoder = deeplab.encoder
        self.decoder = deeplab.decoder
        if not use_bn:
            encoder_name_modules = self.encoder.named_modules()
            self.set_module(encoder_name_modules, self.encoder)
            decoder_name_modules = self.decoder.named_modules()
            self.set_module(decoder_name_modules, self.decoder)

        self.F_encoder = EncoderF(mode=self.mode)
        self.EHigh_encoder = EncoderEhigh(mode=self.mode)
        self.Final_encoder = EncoderFinal(mode=self.mode)

        self.Ref_encoder = EncoderRef()

    def set_module(self, name_modules, root_module):
        for name, m in name_modules:
            if isinstance(m, nn.BatchNorm2d):
                names = name.split(sep='.')
                parents = reduce(getattr, names[:-1], root_module)
                setattr(parents, names[-1], nn.Identity())
            elif isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                else:
                    m.bias = nn.Parameter(torch.zeros(m.out_channels))

    def get_params(self, lr):
        params = [
            {'params': self.encoder_low_enc.parameters(), 'lr': lr},
            {'params': self.encoder_low_dec.parameters(), 'lr': lr},
            {'params': self.encoder_f.parameters(), 'lr': lr},
            {'params': self.encoder_high.parameters(), 'lr': lr},
            {'params': self.encoder_final.parameters(), 'lr': lr},
            {'params': self.encoder_ref.parameters(), 'lr': lr},
        ]
        return params

    def forward(self, x, ref_guidance):
        chan_num = x.shape[1]
        device = x.device
        if chan_num == 3:
            B = x.shape[0]
            H, W = x.shape[2], x.shape[3]
            grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            grid_x = repeat(grid_x, "h w -> b c h w", b=B, c=1).to(device)
            grid_y = repeat(grid_y, "h w -> b c h w", b=B, c=1).to(device)
            x = torch.cat([x, grid_x, grid_y], dim=1)
            chan_num = x.shape[1]
        assert chan_num == 5, "x.shape[1] shoule be 5"
        x_encode = self.encoder(x)
        low_feature = self.decoder(x_encode[-1])
        f_feature = self.F_encoder(low_feature)
        high_feature = self.EHigh_encoder(x)

        if self.ref_fuse_mode == 'high':
            high_ref_feature = self.Ref_encoder(high_feature, ref_guidance)
            output = self.Final_encoder(high_ref_feature, f_feature)
        else:
            raise NotImplementedError
        return output