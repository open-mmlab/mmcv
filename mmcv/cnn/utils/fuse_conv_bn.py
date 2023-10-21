# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


def _fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """Fuse conv / deconv and bn into one module.

    Args:
        conv (nn.Module): Conv / DConv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    if isinstance(conv, nn.Conv2d):
        shape = [conv.out_channels, 1, 1, 1]
    elif isinstance(conv, nn.ConvTranspose2d):
        shape = [1, conv.out_channels, 1, 1]
    else:
        raise NotImplementedError
    conv.weight = nn.Parameter(conv_w * factor.reshape(shape))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_conv_bn(module: nn.Module) -> nn.Module:
    """Recursively fuse conv / dconv and bn in a module.

    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv / dconv layers to save
    computations and simplify network structures.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv / DConv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module
