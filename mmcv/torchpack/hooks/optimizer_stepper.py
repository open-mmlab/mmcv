from torch.nn.utils import clip_grad

from .hook import Hook


class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            clip_grad.clip_grad_norm_(
                filter(lambda p: p.requires_grad, runner.model.parameters()),
                **self.grad_clip)
        runner.optimizer.step()
