# Copyright (c) OpenMMLab. All rights reserved.

from typing import Callable, Optional

import intel_extension_for_pytorch as ipex
import torch

from mmcv.runner import RUNNERS, BaseRunner, EpochBasedRunner, IterBasedRunner
from mmcv.utils import IS_XPU_AVAILABLE


class XPUBaseRunner(BaseRunner):
    """A base runner for XPU.

    This runner has some extra processes for XPU which are shown below:

    1. Parse options for XPU
    2. Optimize pytorch model and optimizer for XPU
    3. Raise errors while encountering illegal usage

    Args:
        model (:obj:`nn.Module`): The model to run.
        batch_processor (callable): A callable method that process a data
            batch. Should be None for XPU runner.
        kwargs (Dict[str, Any], optional): Keyword arguments will be passed to
        ``base_runner.BaseRunner``.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 batch_processor: Optional[Callable] = None,
                 **kwargs) -> None:
        assert hasattr(model, 'train_step') and batch_processor is None,\
            'only support model with train_step'

        super().__init__(model, **kwargs)

        if IS_XPU_AVAILABLE:
            model = self.model.to('xpu')
        else:
            model_device = next(model.parameters()).get_device()
            assert model_device == 'cpu', (
                'XPUBaseRunner supports to optimize models on XPU or CPU, '
                f'got {model_device}.')

        optimize_modules = ipex.optimize(model, optimizer=self.optimizer)
        if self.optimizer is None:
            # no optimizer in eval mode
            self.model = optimize_modules
        else:
            self.model, self.optimizer = optimize_modules


@RUNNERS.register_module()
class XPUEpochBasedRunner(XPUBaseRunner, EpochBasedRunner):
    """Epoch-based Runner for XPU.

    The Inheritance order(MRO) is: XPUEpochBasedRunner -> XPUBaseRunner ->
    EpochBasedRunner -> BaseRunner This runner train models epoch by epoch.
    """
    pass


@RUNNERS.register_module()
class XPUIterBasedRunner(XPUBaseRunner, IterBasedRunner):
    """Iteration-based Runner for XPU.

    The Inheritance order(MRO) is: XPUIterBasedRunner -> XPUBaseRunner ->
    IterBasedRunner -> BaseRunner This runner train models iteration by
    iteration.
    """
    pass
