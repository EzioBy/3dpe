# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import random
import torchvision.transforms as transforms
from PIL import Image

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset_Editing(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset_Editing(Dataset_Editing):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = 512, # Ensure specific resolution, None = highest available.
        ref_img_transform_type   ='mae',
        pose_meta       ='dataset.json',
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.ref_img_transform_type = ref_img_transform_type

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        self._all_fnames = sorted(self._all_fnames)
        self.style_dict = {}
        self.style_list = []


        # Get the image list.
        self._image_fnames = sorted(fname for fname in self._all_fnames if fname.split('.')[-1].lower() in ['jpg', 'png', 'jpeg'])
        # Remove edited image paths and only preserve the input image ones.
        self.items = [item for item in self._image_fnames if '_' not in item]
        # Pure file names without dirname.
        self.item_fnames = [os.path.basename(i) for i in self.items]

        # Dynamically read the styles by dirnames, the key in the dict is ffhq dir num and the value is the style.
        style_dir_list = [i for i in os.listdir(self._path) if os.path.isdir(os.path.join(self._path, i)) and '_' in i]
        for item in style_dir_list:
            item = item.replace('/', '')
            ffhq_index, style = item.split('_', 1)[0], item.split('_', 1)[1]
            if ffhq_index in self.style_dict.keys():
                if not isinstance(self.style_dict[ffhq_index], list):
                    self.style_dict[ffhq_index] = [self.style_dict[ffhq_index]]
                self.style_dict[ffhq_index].append(style)
            else:
                self.style_dict[ffhq_index] = style
            self.style_list.append(style)
        PIL.Image.init()

        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')

        # Load poses (saved in dict).
        self.poses = self._load_raw_poses(open(self._open_file(pose_meta)))
        # Transformation function for MAE validation, which is needed because pre-trained MAE weights are introduced.
        if 'mae' in self.ref_img_transform_type.lower():
            # Resulting 224*224 images normalized with imagenet mean & std.
            self.transform_mae_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224, interpolation=3),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return os.path.join(self._path, fname)
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_poses(self, fp):
        poses_dict = {}
        poses = json.load(fp)['labels']

        fpaths = [fpath for fpath in self.items]
        poses_new = dict(poses)
        poses_new = [
            poses_new[fpath.replace('\\', '/')]
            for fpath in self.items
        ]
        poses_new = np.array(poses_new)
        poses_new = poses_new.astype({1: np.int64, 2: np.float32}[poses_new.ndim])

        for i in range(len(fpaths)):
            poses_dict[fpaths[i]] = poses_new[i]
        return poses_dict

    def get_pose(self, fpath):
        pose = self.poses[fpath]
        return pose.copy()

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        image = np.array(PIL.Image.open(self._open_file(fname)))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    # Read and preprocess the PIL images - return img_tensor.
    def preprocess_img(self, img_path):
        img_pil = Image.open(self._open_file(img_path)).resize((512, 512))
        img_np = np.array(img_pil)
        img_tensor = transforms.ToTensor()(img_pil)
        if img_tensor.shape[1] == 4:  # remove alpha channel
            img_tensor = img_tensor[:, :3, :, :]
        img_tensor = img_tensor * 2 - 1
        return img_np, img_tensor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # The input image.
        image_path = self.items[idx]
        image_name = os.path.basename(image_path)
        image_dir = os.path.dirname(image_path)

        # Get the corresponding style of the chosen input image.
        if isinstance(self.style_dict[image_dir], str):
            target_style = self.style_dict[image_dir]
        elif isinstance(self.style_dict[image_dir], list):
            target_style = random.choice(self.style_dict[image_dir])
        target_style_dir = f'{image_dir}_{target_style}'
        edited_image_path = os.path.join(target_style_dir, image_name)

        # Choose a ref image pair whthin the same directory of the input image.
        ref_image_set = [item for item in self._image_fnames if f'{target_style_dir}/' in item and not item.endswith('/')]
        ref_image_path = random.choice(ref_image_set)

        input_np, input_tensor = self.preprocess_img(image_path)
        ref_pil = Image.open(self._open_file(ref_image_path)).resize((512, 512))
        ref_np, ref_tensor = self.preprocess_img(ref_image_path)
        ref_tensor_mae = self.transform_mae_val(ref_pil)
        gt_np, gt_tensor = self.preprocess_img(edited_image_path)
        # Pose label loaded from the json file.
        pose = self.poses[image_path].copy()
        idx = np.array(idx)

        return_list = [idx, image_path, target_style, pose, input_tensor, ref_tensor, ref_tensor_mae, gt_tensor, input_np, ref_np, gt_np]
        return return_list