import os

cmd = 'python scripts/inference.py \
--model_path ./pretrained_models/3dpe.pt \
--data_dir ./test_imgs/ \
--save_path ./results'

print(cmd)
os.system(cmd)