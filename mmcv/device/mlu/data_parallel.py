# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmcv.parallel import MMDataParallel
from .scatter_gather import scatter_kwargs


class MLUDataParallel(MMDataParallel):
    """The DataParallel module that supports DataContainer.

    MMDataParallel has two main differences with PyTorch DataParallel:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data during both GPU and CPU inference.
    - It implement two more APIs ``train_step()`` and ``val_step()``.

    .. warning::
        MMDataParallel only supports single GPU training, if you need to
        train with multiple GPUs, please use MMDistributedDataParallel
        instead. If you have multiple GPUs and you just want to use
        MMDataParallel, you can set the environment variable
        ``CUDA_VISIBLE_DEVICES=0`` or instantiate ``MMDataParallel`` with
        ``device_ids=[0]``.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        device_ids (list[int]): Device IDS of modules to be scattered to.
            Defaults to None when GPU is not available.
        output_device (str | int): Device ID for output. Defaults to None.
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """

    def __init__(self, *args, dim=0, **kwargs):
        super(MLUDataParallel, self).__init__(*args, dim=dim, **kwargs)
        self.device_ids = [0]
        self.src_device_obj = torch.device('mlu:0')

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
