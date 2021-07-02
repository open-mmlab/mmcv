import numpy as np
import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['contour_expand'])


def contour_expand(kernel_mask, internal_kernel_label, min_kernel_area,
                   kernel_num):
    """Expand kernel contours so that foreground pixels are assigned into
    instances.

    Arguments:
        kernel_mask (np.array or Tensor): The instance kernel mask with
            size hxw.
        internal_kernel_label (np.array or Tensor): The instance internal
            kernel label with size hxw.
        min_kernel_area (int): The minimum kernel area.
        kernel_num (int): The instance kernel number.

    Returns:
        label (np.array or Tensor): The instance index map with size hxw.
    """
    assert isinstance(kernel_mask, (torch.Tensor, np.ndarray))
    assert isinstance(internal_kernel_label, (torch.Tensor, np.ndarray))
    assert isinstance(min_kernel_area, int)
    assert isinstance(kernel_num, int)

    if isinstance(kernel_mask, np.ndarray):
        kernel_mask = torch.from_numpy(kernel_mask)
    if isinstance(internal_kernel_label, np.ndarray):
        internal_kernel_label = torch.from_numpy(internal_kernel_label)

    label = ext_module.contour_expand(kernel_mask, internal_kernel_label,
                                      min_kernel_area, kernel_num)
    return label
