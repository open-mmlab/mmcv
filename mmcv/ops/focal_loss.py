# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'sigmoid_focal_loss_forward', 'sigmoid_focal_loss_backward',
    'softmax_focal_loss_forward', 'softmax_focal_loss_backward'
])


def sigmoid_focal_loss_cpu(input,
                           target,
                           gamma=2.0,
                           alpha=0.25,
                           weight=None,
                           reduction='mean'):
    """Loss used in RetinaNet for dense detection:
    https://arxiv.org/abs/1708.02002.

    Args:
        input (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction with
            shape (N,). Stores the binary classification label for each
            element in inputs. (0 for the negative class and 1 for the
            positive class).
        gamma (float, optional): Exponent of the modulating factor (1 - p_t)
            to balance easy vs hard examples. Default = 2.0
        alpha (float, optional): Weighting factor in range (0,1) tobalance
            positive vs negative examples. Default = 0.25
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        reduction: 'none' | 'mean' | 'sum'
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged.
            'sum': The output will be summed.
            Default = 'mean'

    Returns:
        Loss tensor with the reduction option applied.
    """

    if weight is None:
        weight = torch.ones_like(target)

    p = torch.sigmoid(input)
    target = F.one_hot(target)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    ce_loss = -torch.log(p_t)
    loss = alpha_t * ((1 - p_t)**gamma) * ce_loss
    loss = loss.sum(dim=1)
    loss *= weight
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def softmax_focal_loss_cpu(input,
                           target,
                           gamma=2.0,
                           alpha=0.25,
                           weight=None,
                           reduction='mean'):
    """Focal Loss for multi class.

    Args:
        input (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction with
            shape (N,). Stores the binary classification label for each
            element in inputs. (0 for the negative class and 1 for the
            positive class).
        gamma (float, optional): Exponent of the modulating factor (1 - p_t)
            to balance easy vs hard examples. Default = 2.0
        alpha (float, optional): Weighting factor in range (0,1) tobalance
            positive vs negative examples. Default = 0.25
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        reduction: 'none' | 'mean' | 'sum'
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged.
            'sum': The output will be summed.
            Default = 'mean'

    Returns:
        Loss tensor with the reduction option applied.
    """

    if weight is None:
        weight = torch.ones_like(target)
    channel_stats, _ = torch.max(input, dim=1)

    input_ = input - channel_stats.unsqueeze(1)
    p = F.softmax(input_, dim=1)
    range_n = torch.arange(0, input.size(0))
    p = p[range_n, target]
    loss = -alpha * (1 - p)**gamma * torch.log(p)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


class SigmoidFocalLossFunction(Function):

    @staticmethod
    def symbolic(g, input, target, gamma, alpha, weight, reduction):
        return g.op(
            'mmcv::MMCVSigmoidFocalLoss',
            input,
            target,
            gamma_f=gamma,
            alpha_f=alpha,
            weight_f=weight,
            reduction_s=reduction)

    @staticmethod
    def forward(ctx,
                input,
                target,
                gamma=2.0,
                alpha=0.25,
                weight=None,
                reduction='mean'):

        assert isinstance(target, (torch.LongTensor, torch.cuda.LongTensor))
        assert input.dim() == 2
        assert target.dim() == 1
        assert input.size(0) == target.size(0)
        if weight is None:
            weight = input.new_empty(0)
        else:
            assert weight.dim() == 1
            assert input.size(1) == weight.size(0)
        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        assert reduction in ctx.reduction_dict.keys()

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        output = input.new_zeros(input.size())

        ext_module.sigmoid_focal_loss_forward(
            input, target, weight, output, gamma=ctx.gamma, alpha=ctx.alpha)
        if ctx.reduction == ctx.reduction_dict['mean']:
            output = output.sum() / input.size(0)
        elif ctx.reduction == ctx.reduction_dict['sum']:
            output = output.sum()
        ctx.save_for_backward(input, target, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, target, weight = ctx.saved_tensors

        grad_input = input.new_zeros(input.size())

        ext_module.sigmoid_focal_loss_backward(
            input,
            target,
            weight,
            grad_input,
            gamma=ctx.gamma,
            alpha=ctx.alpha)

        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_input /= input.size(0)
        return grad_input, None, None, None, None, None


sigmoid_focal_loss = SigmoidFocalLossFunction.apply


class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha, weight=None, reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, input, target):
        if torch.cuda.is_available() and input.is_cuda:
            return sigmoid_focal_loss(input, target, self.gamma, self.alpha,
                                      self.weight, self.reduction)
        else:
            return sigmoid_focal_loss_cpu(input, target, self.gamma,
                                          self.alpha, self.weight,
                                          self.reduction)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(gamma={self.gamma}, '
        s += f'alpha={self.alpha}, '
        s += f'reduction={self.reduction})'
        return s


class SoftmaxFocalLossFunction(Function):

    @staticmethod
    def symbolic(g, input, target, gamma, alpha, weight, reduction):
        return g.op(
            'mmcv::MMCVSoftmaxFocalLoss',
            input,
            target,
            gamma_f=gamma,
            alpha_f=alpha,
            weight_f=weight,
            reduction_s=reduction)

    @staticmethod
    def forward(ctx,
                input,
                target,
                gamma=2.0,
                alpha=0.25,
                weight=None,
                reduction='mean'):

        assert isinstance(target, (torch.LongTensor, torch.cuda.LongTensor))
        assert input.dim() == 2
        assert target.dim() == 1
        assert input.size(0) == target.size(0)
        if weight is None:
            weight = input.new_empty(0)
        else:
            assert weight.dim() == 1
            assert input.size(1) == weight.size(0)
        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        assert reduction in ctx.reduction_dict.keys()

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        channel_stats, _ = torch.max(input, dim=1)
        input_softmax = input - channel_stats.unsqueeze(1).expand_as(input)
        input_softmax.exp_()

        channel_stats = input_softmax.sum(dim=1)
        input_softmax /= channel_stats.unsqueeze(1).expand_as(input)

        output = input.new_zeros(input.size(0))
        ext_module.softmax_focal_loss_forward(
            input_softmax,
            target,
            weight,
            output,
            gamma=ctx.gamma,
            alpha=ctx.alpha)

        if ctx.reduction == ctx.reduction_dict['mean']:
            output = output.sum() / input.size(0)
        elif ctx.reduction == ctx.reduction_dict['sum']:
            output = output.sum()
        ctx.save_for_backward(input_softmax, target, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_softmax, target, weight = ctx.saved_tensors
        buff = input_softmax.new_zeros(input_softmax.size(0))
        grad_input = input_softmax.new_zeros(input_softmax.size())

        ext_module.softmax_focal_loss_backward(
            input_softmax,
            target,
            weight,
            buff,
            grad_input,
            gamma=ctx.gamma,
            alpha=ctx.alpha)

        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_input /= input_softmax.size(0)
        return grad_input, None, None, None, None, None


softmax_focal_loss = SoftmaxFocalLossFunction.apply


class SoftmaxFocalLoss(nn.Module):

    def __init__(self, gamma, alpha, weight=None, reduction='mean'):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, input, target):
        if torch.cuda.is_available() and input.is_cuda:
            return softmax_focal_loss(input, target, self.gamma, self.alpha,
                                      self.weight, self.reduction)
        else:
            return softmax_focal_loss_cpu(input, target, self.gamma,
                                          self.alpha, self.weight,
                                          self.reduction)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(gamma={self.gamma}, '
        s += f'alpha={self.alpha}, '
        s += f'reduction={self.reduction})'
        return s
