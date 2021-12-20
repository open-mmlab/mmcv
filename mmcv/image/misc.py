# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import mmcv

try:
    import torch
except ImportError:
    torch = None


def tensor2imgs(tensor, mean=None, std=None, to_rgb=True):
    """Convert tensor to 3-channel images or 1-channel gray images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W). :math:`C` can be either 3 or 1.
        mean (tuple[float], optional): Mean of images. If None, default one
            will be used according to the channel dim of tensor.
            Defaults to None.
        std (tuple[float], optional): Standard deviation of images. If None,
            default one will be used according to the channel dim of tensor.
            Defaults to None.
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            For the tensor with 1 channel, it must be False. Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and tensor.ndim == 4
    assert tensor.size(1) in [1, 3]
    if mean is None:
        mean = (0, ) if tensor.size(1) == 1 else (0, 0, 0)
    if std is None:
        std = (1, ) if tensor.size(1) == 1 else (1, 1, 1)
    assert (tensor.size(1) == len(mean) == len(std) == 3) or \
        (tensor.size(1) == len(mean) == len(std) == 1 and not to_rgb)

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs
