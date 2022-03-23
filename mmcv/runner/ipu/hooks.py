# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.hooks import HOOKS, OptimizerHook, LrUpdaterHook


def wrap_lr_update_hook(lr_hook_class):
    assert issubclass(lr_hook_class, LrUpdaterHook)

    class ipu_lr_hook_class(lr_hook_class):
        def _set_lr(self, runner, *args, **kwargs):
            result = super()._set_lr(runner, *args, **kwargs)
            assert result is None  # _set_lr should return nothing
            runner.model.setOptimizer(runner.optimizer)
    return ipu_lr_hook_class


def wrap_optimizer_hook(optimizer_hook_class,):
    assert optimizer_hook_class == OptimizerHook,\
        'OptimizerHook type used is:{}, not supported now'.format(
            str(optimizer_hook_class))

    class ipu_optimizer_hook_class(OptimizerHook):
        def after_train_iter(self, runner):
            if self.detect_anomalous_params:
                self.detect_anomalous_parameters(
                    runner.outputs['loss'], runner)
            if self.grad_clip is not None:
                raise NotImplementedError('IPU does not support gradient clip')
    return ipu_optimizer_hook_class


if (TORCH_VERSION != 'parrots'
        and digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
    @HOOKS.register_module()
    class IPUFp16OptimizerHook(OptimizerHook):
        """FP16 optimizer hook (using PyTorch's implementation).

        If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
        to take care of the optimization procedure.

        Args:
            loss_scale (float | str | dict): Scale factor configuration.
                If loss_scale is a float, static loss scaling will be used with
                the specified scale. If loss_scale is a string, it must be
                'dynamic', then dynamic loss scaling will be used.
                It can also be a dict containing arguments of GradScalar.
                Defaults to 512. For Pytorch >= 1.6, mmcv uses official
                implementation of GradScaler. If you use a dict version of
                loss_scale to create GradScaler, please refer to:
                https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
                for the parameters.

        Examples:
            >>> loss_scale = dict(
            ...     init_scale=65536.0,
            ...     growth_factor=2.0,
            ...     backoff_factor=0.5,
            ...     growth_interval=2000
            ... )
            >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
        """

        def __init__(self,
                     grad_clip=None,
                     coalesce=True,
                     bucket_size_mb=-1,
                     loss_scale=512.,
                     distributed=True):
            assert grad_clip is None,\
                'IPU mode does not support `grad_clip` currently'
            assert coalesce,\
                'implemented all reduce in distributed training currently'
            assert bucket_size_mb == -1,\
                'no bucket_size_mb can be set in IPU mode'
            self.distributed = distributed
            self._scale_update_param = None
            if loss_scale == 'dynamic':
                raise NotImplementedError(
                    'IPU mode not support dynamic loss scale currently')
            elif isinstance(loss_scale, float):
                self.loss_scale = loss_scale
            elif isinstance(loss_scale, dict):
                raise NotImplementedError(
                    'IPU mode support single scale currently')
            else:
                raise ValueError('loss_scale must be of type float, dict, or '
                                 f'"dynamic", got {loss_scale}')

        def after_train_iter(self, runner):
            pass

else:
    raise RuntimeError('The IPU mode only supports torch1.10 and above')
