import sys
sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn as nn
from segmentation_models_pytorch.decoders.deeplabv3 import DeepLabV3

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from einops import rearrange, repeat
from functools import reduce
from segmentation_models_pytorch.encoders.mix_transformer import OverlapPatchEmbed, Block

from models.eg3d.volumetric_rendering.renderer import ImportanceRenderer, sample_from_planes
from models.eg3d.volumetric_rendering.ray_sampler import RaySampler
from models.eg3d.superresolution import SuperresolutionHybrid8XDC
from models.eg3d.triplane import OSGDecoder
from models.eg3d.networks_stylegan2 import Generator as StyleGAN2Backbone

class TriGenerator(nn.Module):
    '''
        similar to TriplaneGenerator class but lack of renderer
    '''
    def __init__(self,
        z_dim,
        c_dim,
        w_dim,
        img_resolution,
        img_channels,
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self._last_planes = None
        self.rendering_kwargs = rendering_kwargs

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        return planes

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


class TriplaneRenderer(nn.Module):
    def __init__(self, img_resolution, img_channels, rendering_kwargs={}) -> None:
        '''
        Triplane Renderer
            Generate 2D image from triplanes representation
            SuperResolution without stylecode 
            FullyConnected layer
        '''
        super(TriplaneRenderer, self).__init__()
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()

        self.w_dim = 512
        self.const = torch.nn.Parameter(torch.randn([1, 1, self.w_dim]))

        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.superresolution = SuperresolutionHybrid8XDC(32, img_resolution, sr_num_fp16_res=0, sr_antialias=True)
        self.rendering_kwargs = rendering_kwargs
        self.neural_rendering_resolution = 128

    def synthesis(self, planes, c, neural_rendering_resolution=None):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        
        N, _, _ = ray_origins.shape
        
        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, _ = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        const_w_input = self.const.repeat([N, 1, 1])
        sr_image = self.superresolution(rgb_image, feature_image, const_w_input, noise_mode='none')

        return {'image': sr_image, 
                'image_raw': rgb_image, 
                'image_depth': depth_image, 
                'feature_image': feature_image, 
                'planes': planes}

    def sample_density(self, planes, coordinates, directions):
        sampled_features = sample_from_planes(self.renderer.plane_axes, planes, coordinates, padding_mode='zeros', box_warp=self.rendering_kwargs['box_warp'])
        out = self.decoder(sampled_features, directions)
        return out

    def forward(self, planes, c):
        return self.synthesis(planes, c)


class EncoderEhigh(nn.Module):
    def __init__(self, mode="Normal") -> None:
        super(EncoderEhigh, self).__init__()
        self.mode = mode
        self.net = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01)
        ) if self.mode == "Normal" else nn.Sequential(
            # Input is Second layer output of DeeplabV3
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3),  
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            )

        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, img):
        return self.net(img)

class EncoderF(nn.Module):
    def __init__(self, mode="Normal") -> None:
        super(EncoderF, self).__init__()

        self.mode = mode
        self.blocks_num = 5 if self.mode == 'Normal' else 2

        self.patchembed = OverlapPatchEmbed(img_size=64, stride=2, in_chans=256, embed_dim=1024)
        self.blocks = nn.ModuleList([Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1) for _ in range(self.blocks_num)])

        self.upsample = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.UpsamplingBilinear2d(scale_factor=2.0),
        )

        self.net = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2.0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)
        ) if self.mode == "Normal" else nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
        )
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        x, H, W  = self.patchembed(x)
        for i in range(self.blocks_num):
            x = self.blocks[i](x, H, W)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.upsample(x)
        x = self.net(x)
        return x 

class EncoderFinal(nn.Module):
    def __init__(self, mode="Normal") -> None:
        super(EncoderFinal, self).__init__()
        self.mode = mode
        self.net = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
        ) if self.mode == "Normal" else nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
        )

        if self.mode == "Normal":
            self.pathembed = OverlapPatchEmbed(img_size=256, stride=2, in_chans=128, embed_dim=1024)
            self.block = Block(dim=1024, num_heads=2, mlp_ratio=2, sr_ratio=2)
        else:
            self.pathembed = OverlapPatchEmbed(img_size=128, stride=2, in_chans=256, embed_dim=1024)
            self.block = Block(dim=1024, num_heads=2, mlp_ratio=2, sr_ratio=2)
        
        self.pixelShuffel = nn.PixelShuffle(upscale_factor=2)

        self.output_net = nn.Sequential(
            nn.Conv2d(352, 256, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1), 
        ) if self.mode == "Normal" else nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),  
            nn.UpsamplingBilinear2d(scale_factor=2.0),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),  
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01), 
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.01),         
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), 
        )
        self.net.apply(self._init_weights)
        self.output_net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, F_feature):
        xx = self.net(torch.cat([x, F_feature], dim=1))
        xx, H, W = self.pathembed(xx)
        xx = self.block(xx, H, W)
        xx = rearrange(xx, "b (h w) c -> b c h w", h=H, w=W)
        xx = self.pixelShuffel(xx)
        if self.mode == "Normal":
            return self.output_net(torch.cat([xx, F_feature], dim=1))
        else:
            return self.output_net(xx)

class EG3DInvEncoder(nn.Module):
    def __init__(
        self,
        in_channels=5,
        encoder_name="resnet34",
        encoder_depth=3,
        mode="Normal",
        use_bn=False,
    ) -> None:
        super(EG3DInvEncoder, self).__init__()
        self.mode = mode
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

    def set_module(self, name_modules, root_module):
        '''
            Remove BN, Conv add bias
            Original: Conv2d + BN + ReLU (Do not use separate Conv)
            Now: Conv2d(add bias) + ReLU
        '''
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
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.decoder.parameters(), 'lr': lr},
            {'params': self.F_encoder.parameters(), 'lr': lr},
            {'params': self.EHigh_encoder.parameters(), 'lr': lr},
            {'params': self.Final_encoder.parameters(), 'lr': lr},
        ]
        return params

    def forward(self, x):
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
        output = self.Final_encoder(high_feature, f_feature)

        return output

def config_option():
    rendering_options = {
        'image_resolution': 128,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'sr_antialias': True,
    }

    rendering_options.update({
        'depth_resolution': 48, # number of uniform samples to take per ray.
        'depth_resolution_importance': 48, # number of importance samples to take per ray.
        'ray_start': 2.25, # near point along each ray to start taking samples.
        'ray_end': 3.3, # far point along each ray to stop taking samples.
        'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
        'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
        'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
    })

    return rendering_options