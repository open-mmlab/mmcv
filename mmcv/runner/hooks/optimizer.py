# Copyright (c) Open-MMLab. All rights reserved.
import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad

from ..dist_utils import allreduce_grads
from ..fp16_utils import LossScaler, wrap_fp16_model
from .hook import HOOKS, Hook


@HOOKS.register_module()
class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()


@HOOKS.register_module()
class Fp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook.

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float | str | dict): Scale factor multiplied with loss.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of LossScaler.
            Defaults to 512.
    """

    def __init__(self,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 loss_scale=512.,
                 distributed=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        if loss_scale == 'dynamic':
            self.loss_scaler = LossScaler(mode='dynamic')
        elif isinstance(loss_scale, float):
            self.loss_scaler = LossScaler(init_scale=loss_scale, mode='static')
        elif isinstance(loss_scale, dict):
            self.loss_scaler = LossScaler(**loss_scale)
        else:
            raise ValueError('loss_scale must be of type float, dict, or '
                             f'"dynamic", got {loss_scale}')

    def before_run(self, runner):
        """Preparing steps before Mixed Precision Training.

        1. Make a master copy of fp32 weights for optimization.
        2. Convert the main model from fp32 to fp16.
        """
        # keep a copy of fp32 weights
        old_groups = runner.optimizer.param_groups
        runner.optimizer.param_groups = copy.deepcopy(
            runner.optimizer.param_groups)
        state = defaultdict(dict)
        p_map = {
            old_p: p
            for old_p, p in zip(
                chain(*(g['params'] for g in old_groups)),
                chain(*(g['params'] for g in runner.optimizer.param_groups)))
        }
        for k, v in runner.optimizer.state.items():
            state[p_map[k]] = v
        runner.optimizer.state = state
        # convert model to fp16
        wrap_fp16_model(runner.model)

    def copy_grads_to_fp32(self, fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner):
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer `loss_scalar.py`

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients (fp16).
        3. Copy gradients from the model to the fp32 weight copy.
        4. Scale the gradients back and update the fp32 weight copy.
        5. Copy back the params from fp32 weight copy to the fp16 model.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        # scale the loss value
        scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
        scaled_loss.backward()
        # copy fp16 grads in the model to fp32 params in the optimizer

        fp32_weights = []
        for param_group in runner.optimizer.param_groups:
            fp32_weights += param_group['params']
        self.copy_grads_to_fp32(runner.model, fp32_weights)
        # allreduce grads
        if self.distributed:
            allreduce_grads(fp32_weights, self.coalesce, self.bucket_size_mb)

        has_overflow = self.loss_scaler.has_overflow(fp32_weights)
        # if has overflow, skip this iteration
        if not has_overflow:
            # scale the gradients back
            for param in fp32_weights:
                if param.grad is not None:
                    param.grad.div_(self.loss_scaler.loss_scale)
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(fp32_weights)
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            # update fp32 params
            runner.optimizer.step()
            # copy fp32 params to the fp16 model
            self.copy_params_to_fp16(runner.model, fp32_weights)
        self.loss_scaler.update_scale(has_overflow)
        if has_overflow:
            runner.logger.warning('Check overflow, downscale loss scale '
                                  f'to {self.loss_scaler.cur_scale}')
