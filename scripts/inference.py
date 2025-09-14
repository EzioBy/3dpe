import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print('torch version: ', torch.__version__)
print('torch.cuda.is_available: ', torch.cuda.is_available())
from torch.utils.data import DataLoader
import argparse
import numpy as np
from PIL import Image

import sys
sys.path.append(".")
sys.path.append("..")

import imageio
from tqdm import tqdm
from models.triencoder_editing import TriEncoder_Editing
from models.eg3d.shape_utils import convert_sdf_samples_to_ply
from datasets.test_dataset_editing import ImageFolderTestset_Editing


device = "cuda:0"

def gen_rand_pose(device):
    return (torch.rand(1, device=device) - 0.5) * torch.pi/6 + torch.pi/2

def layout_grid(img: np.ndarray, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def generate_video(net, gen_triplanes, savepath, batch_size=1):
    voxel_resolution = 512
    video_kwargs = {}
    video_out = imageio.get_writer(savepath, mode='I', fps=60, codec='libx264', **video_kwargs)
    grid_h, grid_w = 1, 1
    num_keyframes, w_frames = 12, 24

    image_mode = "image"
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range = 0.25
                yaw_range = 0.35
                camera_p = net.eural_to_camera(batch_size, 
                        3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                        3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                        )
                render_res = net.triplane_renderer(gen_triplanes, camera_p)
                img = render_res[image_mode][0]
                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)

        video_out.append_data(layout_grid(torch.stack(imgs), grid_h=1, grid_w=1))
    video_out.close()

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

def extract_mesh(net, gen_triplanes, savepath):
    shape_res = 512
    max_batch = 1000000
    my_plane = gen_triplanes.reshape(1,3,32,256,256)
    samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=net.triplane_generator.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
    samples = samples.to(device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total=samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                sigma = net.triplane_renderer.sample_density(my_plane, samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head])
                sigmas[:, head:head+max_batch] = sigma['sigma']
                head += max_batch
                pbar.update(max_batch)

    sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    # Trim the border of the extracted cube
    pad = int(30 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value

    convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, savepath, level=10)

def multi_view_output(net, gen_triplanes, savepath, batch_size=1, input_np=None, ref_np=None):
    image_mode = "image"
    offset = np.linspace(-np.pi/6, np.pi/6, 3)
    imgs_list = []
    for p in offset:
        camera_p = net.eural_to_camera(batch_size, np.pi/2, np.pi/2 + p)
        render_res = net.triplane_renderer(gen_triplanes, camera_p)
        img = render_res[image_mode][0]
        imgs_list.append((1 + img.clamp(-1,1)).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255)
    # vis ref and input images.
    if ref_np is not None:
        imgs_list.insert(0, ref_np)
    if input_np is not None:
        imgs_list.insert(0, input_np)
    image_list = np.concatenate(imgs_list, axis=1)
    Image.fromarray(image_list.astype(np.uint8)).save(savepath)
    print('Multiview results saved @ ', savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/ssd0/qingyan/3dpe/3dpe.pt', help='pre-trained model path')
    parser.add_argument('--data_dir', type=str, default='/ssd0/qingyan/3dpe/ffhq512_ref_editing_final15/test', help='validation data directory')
    parser.add_argument('--save_path', type=str, default='./results', help='output path')
    parser.add_argument('--save_video', type=bool, default=False, help='If true, save video')
    parser.add_argument('--mode', type=str, default='LT', help='mode of triplane generation', choices=['LT', 'Normal'])
    parser.add_argument('--batch_size', type=int, default=1, help='validation data batchsize')
    args = parser.parse_args()

    model_path = args.model_path
    data_dir = args.data_dir
    outdir = args.save_path
    os.makedirs(outdir, exist_ok=True)
    mode = args.mode
    batch_size = args.batch_size

    val_dataset = ImageFolderTestset_Editing(data_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False)
    ckp_name = os.path.basename(model_path).split('.')[0]

    # load model
    net = TriEncoder_Editing(mode="LT", model_cache_dir='./pretrained_models').to(device)
    print('Net init!')
    ckpt = torch.load(model_path, map_location='cpu')
    print('Ckpt loaded!')
    net.encoder.load_state_dict(ckpt['encoder_state_dict'])
    net.triplane_renderer.load_state_dict(ckpt["renderer_state_dict"])

    count = 0
    for batch in val_dataloader:
        count += 1
        print(f'Sampling #{count} case...')
        idx, image_path, input_tensor, ref_tensor, ref_tensor_mae, input_np, ref_np = batch
        input_tensor = input_tensor.to(device, non_blocking=True)
        ref_tensor = ref_tensor.to(device, non_blocking=True)
        ref_tensor_mae = ref_tensor_mae.to(device, non_blocking=True)
        val_batch_size = input_tensor.size(0)
        edit_triplanes = net.encoder(input_tensor, ref_tensor_mae)
        # Save visualization of img | ref_img | mv_edited_results.
        multi_view_output(net, edit_triplanes,
                          savepath=os.path.join(outdir, f'{count}_mv.jpg'),
                          input_np=input_np[0].cpu().numpy(),
                          ref_np=ref_np[0].cpu().numpy())
        if args.save_video:
            generate_video(net, edit_triplanes, os.path.join(outdir, f'video_{count}.mp4'))
