"""
    Utils for LPIPS models.
"""

from collections import OrderedDict
import os
import torch


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type, cache_dir=''):
    path = os.path.join(cache_dir, f'{net_type}_lin.pth')
    map_loc = None if torch.cuda.is_available() else torch.device('cpu')
    old_state_dict = torch.load(path, map_location=map_loc)

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val
    return new_state_dict