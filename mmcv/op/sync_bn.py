import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from ..utils import ext_loader

ext_module = ext_loader.load_ext('op_ext', [
    'syncbn_forward_step1', 'syncbn_forward_step2', 'syncbn_forward_step3',
    'syncbn_backward_step1', 'syncbn_backward_step2'
])


class SyncBNFunction(Function):
    """
    同步批标准化 Function。

    输入:
        - input(Tensor): 输入
        - running_mean(Tensor): 当前批数据均值
        - running_var(Tensor): 当前批方差
        - weight(Tensor): 权重
        - bias(bool): 偏置
        - momentum(float): 动量，用于 runnning_mean 和 running_var 的计算
        - eps(float): 稳定性数值量
        - group(int): 做 SyncBN 的组数
        - group_size(int): 做 SyncBN 一组的数量

    .. note::
       目前只支持 GPU 端的半精度和全精度的计算。
    """

    @staticmethod
    def forward(self, input, running_mean, running_var, weight, bias, momentum,
                eps, group, group_size):
        self.momentum = momentum
        self.eps = eps
        self.group = group
        self.group_size = group_size

        assert isinstance(
            input, (torch.cuda.HalfTensor, torch.cuda.FloatTensor)), \
            f'only support Half or Float GPU Tensor, but {input.type()}'
        n, c, h, w = input.size()
        mean = torch.empty(c, dtype=torch.float, device='cuda')
        var = torch.empty(c, dtype=torch.float, device='cuda')
        std = torch.empty(c, dtype=torch.float, device='cuda')
        output = torch.empty_like(input)

        # parrots require all tensor parameters at the front and
        # attr parameter passed by keyword
        ext_module.syncbn_forward_step1(input, mean, n=n, c=c, h=h, w=w)
        if self.group_size > 1:
            dist.all_reduce(mean, group=self.group)
            mean /= self.group_size
        ext_module.syncbn_forward_step2(input, mean, var, n=n, c=c, h=h, w=w)
        if self.group_size > 1:
            dist.all_reduce(var, group=self.group)
            var /= self.group_size
        ext_module.syncbn_forward_step3(
            input,
            mean,
            var,
            weight,
            bias,
            running_mean,
            running_var,
            std,
            output,
            n=n,
            c=c,
            h=h,
            w=w,
            group_size=self.group_size,
            eps=self.eps,
            momentum=self.momentum)
        self.save_for_backward(input, mean, weight, std)
        return output

    @staticmethod
    @once_differentiable
    def backward(self, grad_output):
        assert isinstance(
            grad_output, (torch.cuda.HalfTensor, torch.cuda.FloatTensor)), \
            f'only support Half or Float GPU Tensor, but {grad_output.type()}'
        input, mean, weight, std = self.saved_tensors
        n, c, h, w = input.size()
        weight_diff = torch.empty_like(weight)
        bias_diff = torch.empty_like(weight)
        input_diff = torch.empty_like(input)
        # backward step1
        ext_module.syncbn_backward_step1(
            input,
            mean,
            std,
            grad_output,
            weight_diff,
            bias_diff,
            n=n,
            c=c,
            h=h,
            w=w)
        # all reduce
        if self.group_size > 1:
            dist.all_reduce(weight_diff, group=self.group)
            dist.all_reduce(bias_diff, group=self.group)
            weight_diff /= self.group_size
            bias_diff /= self.group_size
        # backward step2
        ext_module.syncbn_backward_step2(
            input,
            mean,
            weight,
            weight_diff,
            bias_diff,
            std,
            grad_output,
            input_diff,
            n=n,
            c=c,
            h=h,
            w=w)
        return input_diff, None, None, weight_diff, bias_diff, \
            None, None, None, None


class SyncBatchNorm2d(Module):
    """
    同步批标准化 Module。

    参数:
        - num_features(Tensor): 特征数
        - eps(float): 稳定性数值量
        - momentum(float): 动量，用于 runnning_mean 和 running_var 的计算
        - affine(bool): affine 参数是否可学习设置
        - track_running_stats(bool): 是否跟踪每轮训练的均值和方差
        - group(int): 做 SyncBN 的组数

    输入:
        - input(Tensor): 特征输入
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 group=dist.group.WORLD):
        super(SyncBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
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
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
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

        if self.training:
            return SyncBNFunction.apply(input, self.running_mean,
                                        self.running_var, self.weight,
                                        self.bias, exponential_average_factor,
                                        self.eps, self.group, self.group_size)
        else:
            return F.batch_norm(input, self.running_mean, self.running_var,
                                self.weight, self.bias, self.training,
                                exponential_average_factor, self.eps)

    def extra_repr(self):
        s = f'{self.num_features}, '
        s += f'eps={self.eps}, '
        s += f'momentum={self.momentum}, '
        s += f'affine={self.affine}, '
        s += f'track_running_stats={self.track_running_stats}, '
        s += f'group_size={self.group_size}'
        return s

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(
                    0, dtype=torch.long)

        super(SyncBatchNorm2d,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
