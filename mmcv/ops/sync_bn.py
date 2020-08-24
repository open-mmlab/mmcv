import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from mmcv.cnn import NORM_LAYERS
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'sync_bn_forward_mean', 'sync_bn_forward_var', 'sync_bn_forward_output',
    'sync_bn_backward_param', 'sync_bn_backward_data'
])


class SyncBatchNormFunction(Function):

    @staticmethod
    def symbolic(g, input, running_mean, running_var, weight, bias, momentum,
                 eps, group, group_size):
        return g.op(
            'MMCVSyncBatchNorm',
            input,
            running_mean,
            running_var,
            weight,
            bias,
            momentum=momentum,
            eps=eps,
            group=group,
            group_size=group_size)

    @staticmethod
    def forward(self, input, running_mean, running_var, weight, bias, momentum,
                eps, group, group_size):
        self.momentum = momentum
        self.eps = eps
        self.group = group
        self.group_size = group_size

        assert isinstance(
                   input, (torch.HalfTensor, torch.FloatTensor,
                           torch.cuda.HalfTensor, torch.cuda.FloatTensor)), \
               f'only support Half or Float Tensor, but {input.type()}'
        output = torch.empty_like(input)
        input3d = input.view(input.size(0), input.size(1), -1)
        output3d = output.view_as(input3d)

        mean = torch.empty(
            input3d.size(1), dtype=torch.float, device=input3d.device)
        var = torch.empty(
            input3d.size(1), dtype=torch.float, device=input3d.device)
        norm = torch.empty_like(
            input3d, dtype=torch.float, device=input3d.device)
        std = torch.empty(
            input3d.size(1), dtype=torch.float, device=input3d.device)

        ext_module.sync_bn_forward_mean(input3d, mean)
        if self.group_size > 1:
            dist.all_reduce(mean, group=self.group)
            mean /= self.group_size
        ext_module.sync_bn_forward_var(input3d, mean, var)
        if self.group_size > 1:
            dist.all_reduce(var, group=self.group)
            var /= self.group_size
        ext_module.sync_bn_forward_output(
            input3d,
            mean,
            var,
            weight,
            bias,
            running_mean,
            running_var,
            norm,
            std,
            output3d,
            eps=self.eps,
            momentum=self.momentum,
            group_size=self.group_size)
        self.save_for_backward(norm, std, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(self, grad_output):
        norm, std, weight = self.saved_tensors
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(weight)
        grad_input = torch.empty_like(grad_output)
        grad_output3d = grad_output.view(
            grad_output.size(0), grad_output.size(1), -1)
        grad_input3d = grad_input.view_as(grad_output3d)
        ext_module.sync_bn_backward_param(grad_output3d, norm, grad_weight,
                                          grad_bias)
        # all reduce
        if self.group_size > 1:
            dist.all_reduce(grad_weight, group=self.group)
            dist.all_reduce(grad_bias, group=self.group)
            grad_weight /= self.group_size
            grad_bias /= self.group_size
        ext_module.sync_bn_backward_data(grad_output3d, weight, grad_weight,
                                         grad_bias, norm, std, grad_input3d)
        return grad_input, None, None, grad_weight, grad_bias, \
            None, None, None, None


@NORM_LAYERS.register_module(name='MMSyncBN')
class SyncBatchNorm(Module):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 group=None):
        super(SyncBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        group = dist.group.WORLD if group is None else group
        self.group = group
        self.group_size = dist.get_world_size(group)
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()  # pytorch use ones_()
            self.bias.data.zero_()

    def forward(self, input):
        if input.dim() < 2:
            raise ValueError(
                f'expected at least 2D input, got {input.dim()}D input')
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or not self.track_running_stats:
            return SyncBatchNormFunction.apply(input, self.running_mean,
                                               self.running_var, self.weight,
                                               self.bias,
                                               exponential_average_factor,
                                               self.eps, self.group,
                                               self.group_size)
        else:
            return F.batch_norm(input, self.running_mean, self.running_var,
                                self.weight, self.bias, False,
                                exponential_average_factor, self.eps)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'({self.num_features}, '
        s += f'eps={self.eps}, '
        s += f'momentum={self.momentum}, '
        s += f'affine={self.affine}, '
        s += f'track_running_stats={self.track_running_stats}, '
        s += f'group_size={self.group_size})'
        return s
