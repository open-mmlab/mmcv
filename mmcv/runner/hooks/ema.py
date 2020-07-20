from ...parallel import is_module_wrapper
from ..hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class EmaHook(Hook):
    r"""Exponential Moving Average Hook

    Use Exponential Moving Average on all parameters of model in training
    process.all parameter has a ema backup, which update by the formula
    as below. The hook must have priority than EvalHook and
    CheckpointSaverHook.

        .. math::

            \text{Xema_{t+1}} = (1 - \text{momentum}) \times
            \text{Xema_{t}} +  \text{momentum} \times X_t

    Args:
        momentum (float): The momentum used for update ema parameter.
            Default to 0.9998.
        interval (int): Update ema parameter every interval iteration.
            Default to 1.
        warm_up (int): During first warm_up steps, we may use smaller momentum
            to update ema parameters more slowly. Default to 100.
        checkpoint (str): The checkpoint path.
    """

    def __init__(self,
                 momentum=0.9998,
                 interval=1,
                 warm_up=100,
                 checkpoint=None):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        self.momentum = momentum**interval
        self.checkpoint = checkpoint

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as named_buffer to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_buffer = {}
        self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        # We warm up the momentum considering the instability  at beginning
        momentum = min(self.momentum,
                       (1 + curr_step) / (self.warm_up + curr_step))
        if curr_step % self.interval != 0:
            return
        for name, parameter in self.model_parameters.items():
            buffer_name = self.param_ema_buffer[name]
            buffer_parameter = self.model_buffers[buffer_name]
            buffer_parameter.mul_(1 - momentum).add_(momentum, parameter.data)

    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self.swap_ema_parameters()

    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's the
        EvalHook."""
        self.swap_ema_parameters()

    def swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
