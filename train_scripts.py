import os

cmd = 'torchrun --nproc_per_node=8 scripts/train.py \
--job_name=your_job \
--model_cache_dir ./pretrained_models \
--train_data_path=/path/to/3dpe_dataset/train \
--val_data_path=/path/to/3dpe_dataset/test \
--exp_dir ./exps \
--device=cuda \
--train_epoch 200 \
--learning_rate_encoder 1e-4 \
--use_gan_loss True \
--add_gan_loss_step 10000 \
--adv_lambda 0.025 \
--lpips_lambda 2.0 \
--depth_lambda 0.5 \
--print_interval 200 \
--image_interval 2000 \
--val_interval 2000 \
--save_interval 2000 \
--batch_size=2 \
--val_batch_size=4'

print(cmd)
os.system(cmd)