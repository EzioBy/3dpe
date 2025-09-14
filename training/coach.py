import os
import matplotlib.pyplot as plt
import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from criteria import id_loss
from criteria.lpips.lpips import LPIPS
from training.ranger import Ranger
from models.triencoder import TriEncoder
from models.triencoder_editing import TriEncoder_Editing
from datasets.dataset_editing import ImageFolderDataset_Editing

class Coach:
    def __init__(self, opts, local_rank):
        self.opts = opts
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{self.local_rank}')
        self.job_name = self.opts.job_name
        self.model_cache_dir = self.opts.model_cache_dir

        if self.opts.use_wandb and self.local_rank == 0:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)

        self.seed_everything(self.opts.seed)

        # Initialize distributed training environment
        self.setup_distributed()

        # Set up directories (only on rank 0 to avoid race conditions)
        if self.local_rank == 0:
            os.makedirs(os.path.join(opts.exp_dir, self.job_name), exist_ok=True)
            log_dir = os.path.join(opts.exp_dir, self.job_name, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            self.logger = SummaryWriter(log_dir=log_dir)
            self.checkpoint_dir = os.path.join(self.opts.exp_dir, self.job_name, 'checkpoints')
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            img_log_dir = os.path.join(self.opts.image_log_path, self.job_name)
            os.makedirs(img_log_dir, exist_ok=True)
        else:
            self.logger = None
            self.checkpoint_dir = None

        self.best_val_loss = None
        self.use_gan_loss = self.opts.use_gan_loss

        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if opts.resume:
            self.resume_path = opts.resume_path

        # Initialize networks
        self.rec_net = TriEncoder(mode="LT", model_cache_dir=self.model_cache_dir).to(self.device)
        self.edit_net = TriEncoder_Editing(mode="LT", model_cache_dir=self.model_cache_dir).to(self.device)

        # Wrap models with DDP
        self.rec_net = nn.parallel.DistributedDataParallel(self.rec_net, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        self.edit_net = nn.parallel.DistributedDataParallel(self.edit_net, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # Initialize loss functions
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        self.lpips_loss = LPIPS(net_type='alex', cache_dir=self.model_cache_dir).to(self.device).eval()
        self.id_loss = id_loss.IDLoss(cache_dir=self.model_cache_dir).to(self.device).eval()
        self.l1loss = nn.L1Loss().to(self.device).eval()

        # Initialize optimizer and scheduler
        self.optimizer_encoder, self.scheduler_encoder = self.configure_optimizers_encoder(self.opts.use_encoder_scheduler)
        if self.use_gan_loss:
            self.D_opt = torch.optim.Adam(self.edit_net.module.D.parameters(), lr=self.opts.lr_D, betas=(0.5, 0.999))

        # Initialize dataset and dataloaders with DistributedSampler
        self.train_dataset = ImageFolderDataset_Editing(self.opts.train_data_path)
        self.val_dataset = ImageFolderDataset_Editing(self.opts.val_data_path)

        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=dist.get_world_size(), rank=self.local_rank, shuffle=True)
        self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=dist.get_world_size(), rank=self.local_rank, shuffle=True)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.opts.batch_size, sampler=self.train_sampler, num_workers=self.opts.workers, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.opts.val_batch_size, sampler=self.val_sampler, num_workers=self.opts.workers, pin_memory=True)

        # Initialize checkpointing variables
        self.global_step = 0

    def setup_distributed(self):
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(self.local_rank)

    def seed_everything(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def gen_rand_pose(self, pitch_range=26, yaw_range=36, cx_range=0.1, cy_range=0.1, fov_range=2.4, mode="yaw"):
        if mode == "yaw":
            return 2 * (torch.rand(1, device=self.device) - 0.5) * (yaw_range / 180 * torch.pi) + torch.pi / 2
        elif mode == "pitch":
            return 2 * (torch.rand(1, device=self.device) - 0.5) * (pitch_range / 180 * torch.pi) + torch.pi / 2
        elif mode == "cx":
            return 2 * (torch.rand(1, device=self.device) - 0.5) * cx_range + 0.5
        elif mode == "cy":
            return 2 * (torch.rand(1, device=self.device) - 0.5) * cy_range + 0.5
        elif mode == "fov":
            return 2 * (torch.rand(1, device=self.device) - 0.5) * fov_range + 18.837

    def load_pretrain_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.rec_net.module.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.rec_net.module.triplane_renderer.load_state_dict(ckpt["renderer_state_dict"])
        self.edit_net.module.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        self.edit_net.module.triplane_renderer.load_state_dict(ckpt["renderer_state_dict"])

        self.edit_net.module.encoder.requires_grad_(True)
        self.edit_net.module.encoder.requires_grad_(True)
        if "step" in ckpt:
            self.global_step = ckpt['step']

    def validate(self):
        # fid50k test, comsuming too much time
        # inception_model = fid50k.load_inception_net(parallel=True)
        pass

    def train_ref_encoder(self):
        self.edit_net.train()
        self.load_pretrain_model(self.resume_path)
        self.edit_net.module.encoder.requires_grad_(True)
        if self.use_gan_loss:
            D = self.edit_net.module.D
            D.requires_grad_(True)

        # The training epoch loop.
        for epoch in range(self.opts.train_epoch):
            self.train_sampler.set_epoch(epoch)
            if self.local_rank == 0:
                print(f'Epoch: {epoch}')
            # The dataset iteration loop.
            for batch in self.train_dataloader:
                idx, image_path, target_style, camera_param, input_tensor, ref_tensor, ref_tensor_mae, gt_tensor, input_np, ref_np, gt_np = batch
                camera_param = camera_param.to(self.device, non_blocking=True)
                input_tensor = input_tensor.to(self.device, non_blocking=True)
                ref_tensor = ref_tensor.to(self.device, non_blocking=True)
                ref_tensor_mae = ref_tensor_mae.to(self.device, non_blocking=True)
                gt_tensor = gt_tensor.to(self.device, non_blocking=True)
                batch_size = input_tensor.size(0)

                self.optimizer_encoder.zero_grad()

                rec_triplanes = self.rec_net.module.encoder(gt_tensor)
                rec_render_res = self.rec_net.module.triplane_renderer(rec_triplanes, camera_param)

                edit_triplanes = self.edit_net.module.encoder(input_tensor, ref_tensor_mae)
                edit_render_res = self.edit_net.module.triplane_renderer(edit_triplanes, camera_param)

                mv_camera_params = self.rec_net.module.eural_to_camera(batch_size,
                                                                self.gen_rand_pose(pitch_range=26, mode='pitch'),
                                                                self.gen_rand_pose(yaw_range=49, mode='yaw'),
                                                                fov_deg=self.gen_rand_pose(mode="fov"),
                                                                cx=self.gen_rand_pose(mode="cx"),
                                                                cy=self.gen_rand_pose(mode="cy"))

                mv_gt_res = self.rec_net.module.triplane_renderer(rec_triplanes, mv_camera_params)

                # loss
                image_loss = self.l1loss(edit_render_res['image'], rec_render_res['image'])
                category_loss = self.id_loss(edit_render_res['image'], rec_render_res['image'], rec_render_res['image'])[0]
                raw_loss = self.l1loss(edit_render_res['image_raw'], rec_render_res['image_raw'])
                depth_loss = self.l1loss(edit_render_res['image_depth'], rec_render_res['image_depth'])
                feature_loss = self.l1loss(edit_render_res['feature_image'], rec_render_res['feature_image'])
                image_lpips_loss = self.lpips_loss(edit_render_res['image'], rec_render_res['image'])
                raw_lpips_loss = self.lpips_loss(edit_render_res['image_raw'], rec_render_res['image_raw'])

                triloss = self.l1loss(edit_triplanes, rec_triplanes)  # triplane loss
                loss = (
                    self.opts.tri_lambda * triloss
                    + self.opts.l1_lambda * image_loss
                    + self.opts.l1_lambda * raw_loss
                    + self.opts.lpips_lambda * image_lpips_loss
                    + self.opts.lpips_lambda * raw_lpips_loss
                    + self.opts.depth_lambda * depth_loss
                    + self.opts.feature_lambda * feature_loss
                    + self.opts.id_lambda * category_loss
                )

                # mv loss
                if self.opts.use_mul_loss:
                    mv_edit_res = self.edit_net.module.triplane_renderer(edit_triplanes, mv_camera_params)
                    mv_image_loss = self.l1loss(mv_edit_res['image'], mv_gt_res['image'])
                    mv_raw_loss = self.l1loss(mv_edit_res['image_raw'], mv_gt_res['image_raw'])
                    mv_depth_loss = self.l1loss(mv_edit_res['image_depth'], mv_gt_res['image_depth'])
                    mv_feature_loss = self.l1loss(mv_edit_res['feature_image'], mv_gt_res['feature_image'])
                    mv_image_lpips_loss = self.lpips_loss(mv_edit_res['image'], mv_gt_res['image'])
                    mv_raw_lpips_loss = self.lpips_loss(mv_edit_res['image_raw'], mv_gt_res['image_raw'])
                    mv_category_loss = self.id_loss(mv_edit_res['image'], mv_gt_res['image'], mv_gt_res['image'])[0]
                    loss += (
                        self.opts.l1_lambda * mv_image_loss
                        + self.opts.l1_lambda * mv_raw_loss
                        + self.opts.lpips_lambda * mv_image_lpips_loss
                        + self.opts.lpips_lambda * mv_raw_lpips_loss
                        + self.opts.depth_lambda * mv_depth_loss
                        + self.opts.feature_lambda * mv_feature_loss
                        + self.opts.id_lambda * mv_category_loss
                    )

                if self.opts.use_gan_loss and self.global_step > self.opts.add_gan_loss_step:
                    r1_gamma = 1
                    self.D_opt.zero_grad()
                    image_ref = ref_tensor.detach().requires_grad_(True)
                    image_res = gt_tensor.detach().requires_grad_(True)
                    gt_logits = D({'image_ref': image_ref, 'image': image_res}, camera_param)

                    loss_Dgt = torch.nn.functional.softplus(-gt_logits).mean()

                    r1_grads = torch.autograd.grad(outputs=[gt_logits.sum()], inputs=[image_res, image_ref],
                                                   create_graph=True, only_inputs=True)
                    r1_grads_image = r1_grads[0]
                    r1_grads_image_raw = r1_grads[1]
                    r1_penalty = r1_grads_image.square().sum([1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
                    loss_Dr1 = (r1_penalty * (r1_gamma / 2))
                    (loss_Dgt + loss_Dr1).mean().backward(retain_graph=True)

                    pred_logits = D(
                        {'image_ref': image_ref.detach(), 'image': edit_render_res['image'].detach()},
                        camera_param)
                    loss_Dgen = torch.nn.functional.softplus(pred_logits).mean()
                    loss_Dgen.backward()
                    self.D_opt.step()

                    # gan loss for generator:
                    logits = D({'image_ref': image_ref, 'image': edit_render_res['image']}, camera_param)
                    loss_gan = torch.nn.functional.softplus(-logits).mean()
                    loss += self.opts.adv_lambda * loss_gan

                loss.backward()
                self.optimizer_encoder.step()
                if self.opts.use_encoder_scheduler:
                    self.scheduler_encoder.step(self.global_step)

                if self.global_step % self.opts.print_interval == 0 and self.local_rank == 0:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f'[{current_time}]')
                    print(f"Step {self.global_step}: loss = {loss.item()}, triloss = {triloss.item()}")
                    print(f"image_loss = {image_loss.item()}, raw_loss = {raw_loss.item()}")
                    print(f"image_lpips_loss = {image_lpips_loss.item()}, raw_lpips_loss = {raw_lpips_loss.item()}")
                    print(f"loss_depth = {depth_loss.item()}, loss_feature = {feature_loss.item()}")
                    self.logger.add_scalar("loss", loss.item(), self.global_step)
                    self.logger.add_scalar("triloss", triloss.item(), self.global_step)
                    self.logger.add_scalar("image_loss", image_loss.item(), self.global_step)
                    self.logger.add_scalar("raw_loss", raw_loss.item(), self.global_step)
                    self.logger.add_scalar("image_lpips_loss", image_lpips_loss.item(), self.global_step)
                    self.logger.add_scalar("raw_lpips_loss", raw_lpips_loss.item(), self.global_step)
                    self.logger.add_scalar("loss_depth", depth_loss.item(), self.global_step)
                    self.logger.add_scalar("loss_feature", feature_loss.item(), self.global_step)
                    self.logger.add_scalar("loss_id", category_loss.item(), self.global_step)

                    if self.opts.use_gan_loss and self.global_step > self.opts.add_gan_loss_step:
                        print(f"loss_gan = {loss_gan.item()}, loss_Dgt = {loss_Dgt.item()}, loss_Dgen = {loss_Dgen.item()}")
                        self.logger.add_scalar("loss_gan", loss_gan.item(), self.global_step)
                        self.logger.add_scalar("loss_Dgt", loss_Dgt.item(), self.global_step)
                        self.logger.add_scalar("loss_Dgen", loss_Dgen.item(), self.global_step)

                    if self.opts.use_mul_loss:
                        print(f"mv_image_loss = {mv_image_loss.item()}, mv_raw_loss = {mv_raw_loss.item()}")
                        print(f"mv_image_lpips_loss = {mv_image_lpips_loss.item()}, mv_raw_lpips_loss = {mv_raw_lpips_loss.item()}")
                        print(f"mv_loss_depth = {mv_depth_loss.item()}, mv_loss_feature = {mv_feature_loss.item()}")
                        self.logger.add_scalar("mv_image_loss", mv_image_loss.item(), self.global_step)
                        self.logger.add_scalar("mv_raw_loss", mv_raw_loss.item(), self.global_step)
                        self.logger.add_scalar("mv_image_lpips_loss", mv_image_lpips_loss.item(), self.global_step)
                        self.logger.add_scalar("mv_raw_lpips_loss", mv_raw_lpips_loss.item(), self.global_step)
                        self.logger.add_scalar("mv_loss_depth", mv_depth_loss.item(), self.global_step)
                        self.logger.add_scalar("mv_loss_feature", mv_feature_loss.item(), self.global_step)
                        self.logger.add_scalar("mv_loss_id", mv_category_loss.item(), self.global_step)

                # Vis training results.
                if self.global_step % self.opts.image_interval == 0 and self.local_rank == 0:
                    with torch.no_grad():
                        vis_edit_render_res = (1 + edit_render_res['image'][0].clamp(-1,1)).detach().cpu().numpy().transpose(1,2,0) / 2 * 255
                        vis_rec_render_res = (1 + rec_render_res['image'][0].clamp(-1,1)).detach().cpu().numpy().transpose(1,2,0) / 2 * 255
                        vis_input = input_np[0].cpu().numpy()
                        vis_ref = ref_np[0].cpu().numpy()
                        vis_gt = gt_np[0].cpu().numpy()
                        plt.imsave(
                            os.path.join(self.opts.image_log_path, self.job_name, f"train_step{str(self.global_step)}.jpg"),
                            np.concatenate([vis_input, vis_ref, vis_gt, vis_rec_render_res, vis_edit_render_res], axis=1).astype(np.uint8)
                        )
                        if self.opts.use_mul_loss:
                            vis_mv_edit_res = (1 + mv_edit_res['image'][0].clamp(-1,1)).detach().cpu().numpy().transpose(1,2,0) / 2 * 255
                            vis_mv_rec_res = (1 + mv_gt_res['image'][0].clamp(-1, 1)).detach().cpu().numpy().transpose(1,2,0) / 2 * 255
                            plt.imsave(
                                os.path.join(self.opts.image_log_path, self.job_name, f"train_mv_step{str(self.global_step)}.jpg"),
                                np.concatenate([vis_input, vis_ref, vis_gt, vis_mv_rec_res, vis_mv_edit_res], axis=1).astype(np.uint8)
                            )

                # Validation.
                if self.global_step % self.opts.val_interval == 0 and self.local_rank == 0:
                    with torch.no_grad():
                        for batch in self.val_dataloader:
                            idx, image_path, target_style, camera_param, input_tensor, ref_tensor, ref_tensor_mae, gt_tensor, input_np, ref_np, gt_np = batch
                            camera_param = camera_param.to(self.device, non_blocking=True)
                            input_tensor = input_tensor.to(self.device, non_blocking=True)
                            ref_tensor_mae = ref_tensor_mae.to(self.device, non_blocking=True)
                            gt_tensor = gt_tensor.to(self.device, non_blocking=True)
                            val_batch_size = input_tensor.size(0)

                            rec_triplanes = self.rec_net.module.encoder(gt_tensor)
                            rec_render_res = self.rec_net.module.triplane_renderer(rec_triplanes, camera_param)

                            edit_triplanes = self.edit_net.module.encoder(input_tensor, ref_tensor_mae)
                            edit_render_res = self.edit_net.module.triplane_renderer(edit_triplanes, camera_param)

                            mv_camera_params = self.rec_net.module.eural_to_camera(self.opts.val_batch_size,
                                                                                   self.gen_rand_pose(pitch_range=26,
                                                                                                      mode='pitch'),
                                                                                   self.gen_rand_pose(yaw_range=49,
                                                                                                      mode='yaw'),
                                                                                   fov_deg=self.gen_rand_pose(mode="fov"),
                                                                                   cx=self.gen_rand_pose(mode="cx"),
                                                                                   cy=self.gen_rand_pose(mode="cy"))

                            mv_gt_res = self.rec_net.module.triplane_renderer(rec_triplanes, mv_camera_params)
                            mv_edit_res = self.edit_net.module.triplane_renderer(edit_triplanes, mv_camera_params)

                            # loss
                            image_lpips_loss = self.lpips_loss(edit_render_res['image'], rec_render_res['image'])
                            mv_image_lpips_loss = self.lpips_loss(mv_edit_res['image'], mv_gt_res['image'])
                            print(f"Val @ Step {self.global_step}: lpips = {image_lpips_loss.item()}, mv lpips = {mv_image_lpips_loss.item()}")

                            # Visualizing val results.
                            for i in range(val_batch_size):
                                vis_input = input_np[i].cpu().numpy()
                                vis_ref = ref_np[i].cpu().numpy()
                                vis_gt = gt_np[i].cpu().numpy()
                                vis_edit_render_res = (1 + edit_render_res['image'][i].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255
                                vis_rec_render_res = (1 + rec_render_res['image'][i].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255
                                vis_temp = np.concatenate([vis_input, vis_ref, vis_gt, vis_rec_render_res, vis_edit_render_res], axis=1).astype(np.uint8)
                                if i == 0:
                                    vis_batch = vis_temp
                                else:
                                    vis_batch = np.concatenate([vis_batch, vis_temp], axis=0).astype(np.uint8)

                                vis_mv_edit_res = (1 + mv_edit_res['image'][i].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255
                                vis_mv_rec_res = (1 + mv_gt_res['image'][i].clamp(-1,1)).detach().cpu().numpy().transpose(1, 2, 0) / 2 * 255
                                vis_mv_temp = np.concatenate([vis_input, vis_ref, vis_gt, vis_mv_rec_res, vis_mv_edit_res], axis=1).astype(np.uint8)
                                if i == 0:
                                    vis_mv_batch = vis_mv_temp
                                else:
                                    vis_mv_batch = np.concatenate([vis_mv_batch, vis_mv_temp], axis=0).astype(np.uint8)

                            plt.imsave(os.path.join(self.opts.image_log_path, self.job_name, f"val_step{str(self.global_step)}.jpg"), vis_batch)
                            plt.imsave(os.path.join(self.opts.image_log_path, self.job_name, f"val_mv_step{str(self.global_step)}.jpg"), vis_mv_batch)

                            break

                if self.global_step % self.opts.save_interval == 0 and self.local_rank == 0:
                    self.checkpoint_me(is_best=False)

                self.global_step += 1

    def checkpoint_me(self, is_best):
        if self.local_rank != 0:
            return
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)

    def configure_optimizers_encoder(self, add_scheduler=False):
        params = list(self.edit_net.module.encoder.Final_encoder.parameters()) + list(self.edit_net.module.encoder.Ref_encoder.parameters())
        if self.opts.optim_name_encoder == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_encoder)
        elif self.opts.optim_name_encoder == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.opts.learning_rate_encoder)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate_encoder)

        if add_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            return optimizer, scheduler
        else:
            return optimizer, None

    def __get_save_dict(self):
        save_dict = {
            'encoder_state_dict': self.edit_net.module.encoder.state_dict(),
            'renderer_state_dict': self.edit_net.module.triplane_renderer.state_dict(),
            'discriminator_state': self.edit_net.module.D.state_dict(),
            'opts': vars(self.opts),
            'best_val_loss': self.best_val_loss,
            'step': self.global_step,
            'encoder_optimizer': self.optimizer_encoder.state_dict(),
        }
        return save_dict