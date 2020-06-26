# Modified from flops-counter.pytorch by Vladislav Sovrasov
# original repo: https://github.com/sovrasov/flops-counter.pytorch

# MIT License

# Copyright (c) 2018 Vladislav Sovrasov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys

import numpy as np
import torch
import torch.nn as nn

from mmcv.utils.parrots_wrapper import (_AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd,
                                        _AvgPoolNd, _BatchNorm, _ConvNd,
                                        _ConvTransposeMixin, _MaxPoolNd)


def get_model_complexity_info(model,
                              input_shape,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None,
                              ost=sys.stdout):
    """Get complexity information of a model.

    This method can calculate FLOPs and parameter counts of a model with
    corresponding input shape. It can also print complexity information for
    each layer in a model.

    Args:
        model (nn.Module): The model for complexity calculation.
        input_shape (tuple): Input shape used for calculation.
        print_per_layer_stat (bool): Whether to print complexity information
            for each layer in a model. Default: True.
        as_strings (bool): Output FLOPs and params counts in a string form.
            Default: True.
        input_constructor (None | callable): If specified, it takes a callable
            method that generates input. otherwise, it will generate a random
            tensor with input shape to calculate FLOPs. Default: None.
        ost (stream): same as :func:`print`. Default: sys.stdout.

    Returns:
        str | float: If `as_strings` is set to True, it will return FLOPs in a
            string format. otherwise, it will return that in a float number
            format.
        str | float: If `as_strings` is set to True, it will return parameter
            counts in a string format. otherwise, it will return that in a
            float number format.
    """
    assert type(input_shape) is tuple
    assert len(input_shape) >= 2
    flops_model = add_flops_counting_methods(model)
    flops_model.eval().start_flops_count()
    if input_constructor:
        input = input_constructor(input_shape)
        _ = flops_model(**input)
    else:
        batch = torch.ones(()).new_empty(
            (1, *input_shape),
            dtype=next(flops_model.parameters()).dtype,
            device=next(flops_model.parameters()).device)
        flops_model(batch)

    if print_per_layer_stat:
        print_model_with_flops(flops_model, ost=ost)
    flops_count = flops_model.compute_average_flops_cost()
    params_count = get_model_parameters_number(flops_model)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units='GMac', precision=2):
    """Convert FLOPs number into a string.

    Args:
        flops (float): FLOPs number to be converted.
        units (None | str): Converted FLOPs units. Options are None, "GMac",
            "MMac", "KMac", "Mac". If set to None, it will automatically
            choose the most suitable unit for FLOPs. Default: 'GMac'.
        precision (int): Digit number after the decimal point. Default: 2.

    Return:
        str: The converted FLOPs number.
    """
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num):
    """Convert parameter number into a string.

    Args:
        params_num (float): Parameter number to be converted.

    Returns:
        str: The converted parameter number.

    Example:
        >>> params_to_string(1e9)
        '1000.0 M'
        >>> params_to_string(2e5)
        '200.0 k'
        >>> params_to_string(3e-9)
        '3e-09'
    """
    if params_num // 10**6 > 0:
        return str(round(params_num / 10**6, 2)) + ' M'
    elif params_num // 10**3:
        return str(round(params_num / 10**3, 2)) + ' k'
    else:
        return str(params_num)


def print_model_with_flops(model, units='GMac', precision=3, ost=sys.stdout):
    """Print a model with FLOPs for each layer.

    Args:
        model (nn.Module): The model to be printed.
        units (None | str): Converted FLOPs units. Default: 'GMac'.
        precision (int): Digit number after the decimal point. Default: 3.
        ost (stream): same as :func:`print`. Default: sys.stdout.

    Example:
        >>> class ExampleModel(nn.Module):

        >>> def __init__(self):
        >>>     super().__init__()
        >>>     self.conv1 = nn.Conv2d(3, 8, 3)
        >>>     self.conv2 = nn.Conv2d(8, 256, 3)
        >>>     self.conv3 = nn.Conv2d(256, 8, 3)
        >>>     self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        >>>     self.flatten = nn.Flatten()
        >>>     self.fc = nn.Linear(8, 1)

        >>> def forward(self, x):
        >>>     x = self.conv1(x)
        >>>     x = self.conv2(x)
        >>>     x = self.conv3(x)
        >>>     x = self.avg_pool(x)
        >>>     x = self.flatten(x)
        >>>     x = self.fc(x)
        >>>     return x

        >>> model = ExampleModel()
        >>> print_model_with_flops(model)
        ExampleModel(
          0.005 GMac, 100.000% MACs,
          (conv1): Conv2d(0.0 GMac, 0.959% MACs, 3, 8, kernel_size=(3, 3), stride=(1, 1))   # noqa: E501
          (conv2): Conv2d(0.003 GMac, 58.760% MACs, 8, 256, kernel_size=(3, 3), stride=(1, 1))
          (conv3): Conv2d(0.002 GMac, 40.264% MACs, 256, 8, kernel_size=(3, 3), stride=(1, 1))
          (avg_pool): AdaptiveAvgPool2d(0.0 GMac, 0.017% MACs, output_size=(1, 1))
          (flatten): Flatten(0.0 GMac, 0.000% MACs, )
          (fc): Linear(0.0 GMac, 0.000% MACs, in_features=8, out_features=1, bias=True)
        )
    """
    total_flops = model.compute_average_flops_cost()

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([
            flops_to_string(
                accumulated_flops_cost, units=units, precision=precision),
            f'{accumulated_flops_cost / total_flops:.3%} MACs',
            self.original_extra_repr()
        ])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model, file=ost)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    """Calculate parameter number of a model.

    Args:
        model (nn.module): The model for parameter number calculation.

    Returns:
        float: Parameter number of the model.
    """
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(
        net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(
        net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(
        net_main_module)
    net_main_module.compute_average_flops_cost = \
        compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    # Adding variables necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self):
    """Compute average FLOPs cost.

    A method to compute average FLOPs cost, which will be available after
    `add_flops_counting_methods()` is called on a desired net object.

    Returns:
        float: Current mean flops consumption per image.
    """
    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__

    return flops_sum / batches_count


def start_flops_count(self):
    """Activate the computation of mean flops consumption per image.

    A method to activate the computation of mean flops consumption per image.
    which will be available after `add_flops_counting_methods()` is called on
    a desired net object. It should be called before running the network.
    """
    add_batch_counter_hook_function(self)
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """Stop computing the mean flops consumption per image.

    A method to stop computing the mean flops consumption per image, which
    will be available after `add_flops_counting_methods()` is called on a
    desired net object. It can be called to pause the computation whenever.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """Reset statistics computed so far.

    A method to Reset computed statistics, which will be available
    after `add_flops_counting_methods()` is called on a desired net object.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module, mask):

    def add_flops_mask_func(module):
        if isinstance(module, torch.nn.Conv2d):
            module.__mask__ = mask

    module.apply(add_flops_mask_func)


def remove_flops_mask(module):
    module.apply(add_flops_mask_variable_or_reset)


def is_supported_instance(module):
    for mod in hook_mapping:
        if issubclass(type(module), mod):
            return True
    return False


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    batch_size = input.shape[0]
    module.__flops__ += int(batch_size * input.shape[1] * output.shape[1])


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def bn_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def gn_flops_counter_hook(module, input, output):
    elems = np.prod(input[0].shape)
    # there is no precise FLOPs estimation of computing mean and variance,
    # and we just set it 2 * elems: half muladds for computing
    # means and half for computing vars
    batch_flops = 3 * elems
    if module.affine:
        batch_flops += elems
    module.__flops__ += int(batch_flops)


def deconv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = np.prod(
        kernel_dims) * in_channels * filters_per_channel

    active_elements_count = batch_size * np.prod(output_dims)
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if conv_module.bias is not None:
        output_dims = list(output.shape[2:])
        bias_flops = out_channels * batch_size * np.prod(output_dims)
    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = np.prod(
        kernel_dims) * in_channels * filters_per_channel

    active_elements_count = batch_size * np.prod(output_dims)
    if conv_module.__mask__ is not None:
        # (b, 1, h, w)
        output_height, output_width = output.shape[2:]
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height,
                                                 output_width)
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


hook_mapping = {
    # deconv
    _ConvTransposeMixin: deconv_flops_counter_hook,
    # conv
    _ConvNd: conv_flops_counter_hook,
    # fc
    nn.Linear: linear_flops_counter_hook,
    # pooling
    _AvgPoolNd: pool_flops_counter_hook,
    _MaxPoolNd: pool_flops_counter_hook,
    _AdaptiveAvgPoolNd: pool_flops_counter_hook,
    _AdaptiveMaxPoolNd: pool_flops_counter_hook,
    # activation
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # normalization
    _BatchNorm: bn_flops_counter_hook,
    nn.GroupNorm: gn_flops_counter_hook,
    # upsample
    nn.Upsample: upsample_flops_counter_hook,
}


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        print('Warning! No positional inputs found for a module, '
              'assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        for mod_type, counter_hook in hook_mapping.items():
            if issubclass(type(module), mod_type):
                handle = module.register_forward_hook(counter_hook)
                break

        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


# --- Masked flops counting
# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):
    if is_supported_instance(module):
        module.__mask__ = None
