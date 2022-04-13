# Copyright (c) OpenMMLab. All rights reserved.
import torch


def scatter(input, devices, streams=None):
    """Scatters tensor to MLU."""
    if isinstance(input, list):
        outputs = [scatter(_input, devices, None) for _input in input]
        return outputs
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        if devices != [-1]:
            output = output.to('mlu')
        return output
    else:
        raise Exception(f'Unknown type {type(input)}.')


def get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        if input.device.type == 'mlu':
            input_device = 'mlu'
            return input_device
        return -1
    else:
        raise Exception(f'Unknown type {type(input)}.')


class Scatter:

    @staticmethod
    def forward(target_mlus, input):
        outputs = scatter(input, target_mlus)
        return tuple(outputs) if isinstance(outputs, list) else (outputs, )
