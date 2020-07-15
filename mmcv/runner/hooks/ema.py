import logging

from ...parallel import is_module_wrapper
from ..hooks.hook import HOOKS, Hook

logger = logging.getLogger('global')


@HOOKS.register_module()
class EmaHook(Hook):
    r""" Use Exponential Moving Average on all parameters of model in training
        process.all parameter has a ema backup, which update by the formula
        as below.

        .. math::

            \text{Xema_{t+1}} = (1 - \text{momentum}) \times
            \text{Xema_{t}} +  \text{momentum} \times X_t

    Args:
        momentum (float): used for update ema parameter.
        interval (int): update ema parameter every interval iteration
        warm_up (int): during first warm_up steps, we may use smaller momentum
            to update ema parameters more slowly.
        resume_from (str): the checkpoint path
    """

    def __init__(self,
                 momentum=0.1,
                 interval=1,
                 warm_up=100,
                 resume_from=None):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        self.momentum = momentum**interval
        self.resume_from = resume_from

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as named_buffer to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.parameter_emabuffer = {}
        self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.parameter_emabuffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        if self.resume_from is not None:
            runner.resume(self.resume_from)

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        momentum = min(self.momentum,
                       (1 + curr_step) / (self.warm_up + curr_step))
        if curr_step % self.interval != 0:
            return
        for name, parameter in self.model_parameters.items():
            buffer_name = self.parameter_emabuffer[name]
            buffer_parameter = self.model_buffers[buffer_name]
            buffer_parameter.mul_(1 - momentum).add_(momentum, parameter.data)

    def after_train_epoch(self, runner):
        self.swap_ema_parameters()

    def before_train_epoch(self, runner):
        self.swap_ema_parameters()

    def before_val_epoch(self, runner):
        self.swap_ema_parameters()

    def after_val_epoch(self, runner):
        self.swap_ema_parameters()

    def swap_ema_parameters(self, ):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.parameter_emabuffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
