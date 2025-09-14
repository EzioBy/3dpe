import sys

sys.path.append(".")
sys.path.append("..")
import os
import math
import torch
from torch import nn
from torch_utils import misc
from models.model_ref import Live3DEditingEncoder
from models.model import TriGenerator, TriplaneRenderer
from models.eg3d.triplane import TriPlaneGenerator
from models.eg3d.camera_utils import FOV_to_intrinsics, LookAtPoseSampler
from models.eg3d.dual_discriminator import DualDiscriminator_Editing


class TriEncoder_Editing(nn.Module):
    def __init__(self, mode="LT", model_cache_dir='./pretrained_models', load_D=False):
        super(TriEncoder_Editing, self).__init__()
        self.device = "cuda"
        self.mode = mode
        self.model_cache_dir = model_cache_dir
        self.load_D = load_D
        self.set_encoder()
        self.set_eg3d()

    def set_eg3d(self):
        rendering_options = {
            'image_resolution': 128,
            'disparity_space_sampling': False,
            'clamp_mode': 'softplus',
            'sr_antialias': True,
        }
        rendering_options.update({
            'depth_resolution': 48,  # number of uniform samples to take per ray.
            'depth_resolution_importance': 48,  # number of importance samples to take per ray.
            'ray_start': 2.25,  # near point along each ray to start taking samples.
            'ray_end': 3.3,  # far point along each ray to stop taking samples.
            'box_warp': 1,
            # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7,  # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0.2],  # used only in the visualizer to control center of camera rotation.
        })

        img_resolution, img_channels = 512, 3

        eg3d_model_path = os.path.join(self.model_cache_dir, 'ffhqrebalanced512-128.pth')
        with open(eg3d_model_path, 'rb') as f:
            resume_data = torch.load(f)
            self.G = TriPlaneGenerator(*resume_data["G_init_args"],
                                       **resume_data["G_init_kwargs"]).eval().requires_grad_(False).to(self.device)
            self.G.load_state_dict(resume_data['G'])
            self.G.neural_rendering_resolution = resume_data["G_neural_rendering_resolution"]
            self.G.rendering_kwargs = resume_data["G_renderinig_kwargs"]

            self.D = DualDiscriminator_Editing(*resume_data["D_init_args"],
                                               **resume_data["D_init_kwargs"]).eval().requires_grad_(False).to(self.device)
            if self.load_D:
                self.D.load_state_dict(resume_data['D'])

        self.triplane_generator = TriGenerator(*resume_data["G_init_args"],
                                               **resume_data["G_init_kwargs"]).eval().requires_grad_(False).to(self.device)
        misc.copy_params_and_buffers(self.G.backbone, self.triplane_generator.backbone, require_all=True)
        self.triplane_renderer = TriplaneRenderer(img_resolution=img_resolution, img_channels=img_channels,
                                                  rendering_kwargs=rendering_options).eval().requires_grad_(False).to(self.device)
        misc.copy_params_and_buffers(self.G.decoder, self.triplane_renderer.decoder, require_all=True)
        misc.copy_params_and_buffers(self.G.superresolution, self.triplane_renderer.superresolution, require_all=True)
        self.triplane_renderer.neural_rendering_resolution = self.G.neural_rendering_resolution
        self.triplane_generator.rendering_kwargs = self.G.rendering_kwargs
        self.triplane_renderer.rendering_kwargs = self.G.rendering_kwargs

    @staticmethod
    def FOV_cxy_to_intrinsics(fov_deg, cx, cy, device='cuda'):
        """Converts FOV and image center to a 3x3 camera intrinsic matrix."""
        focal_length = float(1 / (math.tan(fov_deg * 3.14159 / 360) * 1.414))
        intrinsics = torch.tensor([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], device=device)
        return intrinsics

    def eural_to_camera(self, batch_size, pitch, yaw, fov_deg=18.837, cx=0.5, cy=0.5):
        intrinsics = self.FOV_cxy_to_intrinsics(fov_deg, cx, cy, device=self.device).reshape(-1, 9).repeat(batch_size,
                                                                                                           1)
        cam_pivot = torch.tensor(self.triplane_renderer.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0.2]),
                                 device=self.device)
        cam_radius = self.triplane_renderer.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(yaw, pitch, cam_pivot, radius=cam_radius, batch_size=batch_size,
                                                  device=self.device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics], 1)
        return camera_params

    @torch.no_grad()
    def sample_triplane(self, batch_size, pitch, yaw, fov_deg=18.837, cx=0.5, cy=0.5):
        z = torch.randn((batch_size, self.G.z_dim)).to(self.device)
        truncation_psi = 1
        truncation_cutoff = 14

        camera_params = self.eural_to_camera(batch_size, pitch, yaw, fov_deg, cx, cy)
        ws = self.triplane_generator.mapping(z, camera_params, truncation_psi=truncation_psi,
                                             truncation_cutoff=truncation_cutoff)
        triplanes = self.triplane_generator.synthesis(ws)
        gt = self.G.synthesis(ws, camera_params)
        return triplanes, gt, camera_params, ws

    @torch.no_grad()
    def render_from_pretrain(self, batch_size, pitch, yaw, ws, fov_deg=18.837, cx=0.5, cy=0.5):
        camera_params = self.eural_to_camera(batch_size, pitch, yaw, fov_deg, cx, cy)
        gt = self.G.synthesis(ws, camera_params)
        return gt, camera_params

    @torch.no_grad()
    def sample_from_synthesis(self, batch_size, pitch, yaw):
        z = torch.randn((batch_size, self.G.z_dim)).to(self.device)

        cam_pivot = torch.tensor(self.G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0.2]), device=self.device)
        cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)
        truncation_psi = 1
        truncation_cutoff = 14
        fov_deg = 18.837
        intrinsics = FOV_to_intrinsics(fov_deg, device=self.device).reshape(-1, 9).repeat(batch_size, 1)
        cam2world_pose = LookAtPoseSampler.sample(yaw, pitch, cam_pivot, radius=cam_radius, batch_size=batch_size,
                                                  device=self.device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(yaw, pitch, cam_pivot, radius=cam_radius,
                                                               batch_size=batch_size, device=self.device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics], 1)
        ws = self.G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.G.synthesis(ws, camera_params)['image']
        return img

    def set_encoder(self):
        self.encoder = Live3DEditingEncoder(in_channels=5, mode='LT', encoder_name="resnet34", encoder_depth=3,
                                            use_bn=False)

    def forward(self, x, ref, c):
        x = self.encoder(x, ref)
        x = x.contiguous()
        x = self.triplane_renderer(x, c)
        return x
