# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F


def slice_statically(tensor, indices, dim):
    """Slice tensor according given dimension and indices.

    Args:
        tensor (Tensor): The tensor to be sliced
        indices (Tensor): The indices on the specified dimension,
            the negative indices are seemed as 0.
        dim (int): Target dimension
    Returns:
        result (Tensor): Sliced tensor
    """
    indices = indices.clamp(0)
    selectors = F.one_hot(
        indices.long(), num_classes=tensor.shape[dim]).float()
    if dim != 0:
        tensor = tensor.transpose(0, dim)
    org_type = tensor.dtype
    tensor = tensor.float()
    org_shape = list(tensor.shape)
    tensor = tensor.reshape(org_shape[0], -1)
    result = torch.matmul(selectors, tensor)
    org_shape[0] = selectors.shape[0]
    result = result.reshape(org_shape)
    if dim != 0:
        result = result.transpose(dim, 0)
    return result.to(org_type)


class IPUIdentity(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        add_value = x.new_tensor(1e-6)
        x = x + add_value
        x = x - add_value
        return x
