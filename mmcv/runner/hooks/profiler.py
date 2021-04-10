# Copyright (c) Open-MMLab. All rights reserved.
import warnings
from typing import Callable, List, Optional, Union

import torch

from ..dist_utils import master_only
from .hook import HOOKS, Hook


@HOOKS.register_module()
class ProfilerHook(Hook):
    """Profiler context manager.

    PyTorch Profiler is a tool that allows the collection of the performance
    metrics during the training and inference. More details on Profiler can
    be found at
    https://pytorch.org/docs/1.8.1/profiler.html#torch.profiler.profile

    Args:
        activities (list[str]): List of activity groups (CPU, CUDA) to use in
            profiling.
            Default: ['cpu', 'cuda'].
        schedule (dict, optional): Config of generating the callable schedule.
            if schedule is None, profiler will not add step markers into the
            trace and table view.
            Default: None.
        on_trace_ready (callable, dict): Either a handler or a dict of generate
            handler.
            Default: None.
        record_shapes (bool): Save information about operator's input shapes.
            Default: False.
        profile_memory (bool): Track tensor memory allocation/deallocation.
            Default: False.
        with_stack (bool): Record source information (file and line number)
            for the ops.
            Default: False.
        with_flops (bool): Use formula to estimate the FLOPS of specific
            operators (matrix multiplication and 2D convolution).
            Default: False.
        json_trace_path (str, optional): Exports the collected trace in Chrome
            JSON format.
            Default: None.

    Example:
        >>> runner = ... # instantiate a Runner
        >>> schedule = dict(wait=1, warmup=1, active=2)
        >>> # tensorboard trace or log trace
        >>> trace_config = dict(type='tb_trace', dir_name='work_dir')
        >>> trace_config = dict(type='log_trace',
        >>>                     sort_by='self_cuda_time_total',
        >>>                     row_limit=-1)
        >>> profiler_config = dict(schedule=schedule,
        >>>                        on_trace_ready=trace_config)
        >>> runner.register_profiler_hook(profiler_config)
        >>> runner.run(data_loaders=[trainloader], workflow=[('train', 1)])
    """

    def __init__(self,
                 activities: List[str] = ['cpu', 'cuda'],
                 schedule: Optional[dict] = None,
                 on_trace_ready: Optional[Union[Callable, dict]] = None,
                 record_shapes: bool = False,
                 profile_memory: bool = False,
                 with_stack: bool = False,
                 with_flops: bool = False,
                 json_trace_path: Optional[str] = None) -> None:
        try:
            from torch import profiler  # torch version >= 1.8.0
        except ImportError:
            raise ImportError('profiler is the new feature of torch1.8.0, '
                              f'but your verison is {torch.__version__}')

        if not isinstance(activities, list):
            raise ValueError(
                f'activities should be list, but got {type(activities)}')
        self.activities = []
        for activity in activities:
            activity = activity.lower()
            if activity == 'cpu':
                self.activities.append(profiler.ProfilerActivity.CPU)
            elif activity == 'cuda':
                self.activities.append(profiler.ProfilerActivity.CUDA)
            else:
                raise ValueError(
                    f'activity should be "cpu" or "cuda", but got {activity}')

        if schedule is not None:
            self.schedule = profiler.schedule(**schedule)
        else:
            self.schedule = None

        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.json_trace_path = json_trace_path

    @master_only
    def before_run(self, runner):
        if callable(self.on_trace_ready):  # handler
            _on_trace_ready = self.on_trace_ready
        elif isinstance(self.on_trace_ready, dict):  # config of handler
            trace_type = self.on_trace_ready.pop('type')  # log_trace handler
            if trace_type == 'log_trace':

                def _log_hanlder(prof):
                    print(prof.key_averages().table(**self.on_trace_ready))

                _on_trace_ready = _log_hanlder
            elif trace_type == 'tb_trace':  # tensorboard_trace handler
                try:
                    import torch_tb_profiler  # noqa: F401
                except ImportError:
                    raise ImportError('please run "pip install '
                                      'torch-tb-profiler" to install '
                                      'torch_tb_profiler')
                _on_trace_ready = torch.profiler.tensorboard_trace_handler(
                    **self.on_trace_ready)
            else:
                raise ValueError('trace_type should be "log_trace" or '
                                 f'"tb_trace", but got {trace_type}')
        elif self.on_trace_ready is None:
            _on_trace_ready = None  # type: ignore
        else:
            raise ValueError('on_trace_ready should be handler, dict or None, '
                             f'but got {type(self.on_trace_ready)}')

        if runner.max_epochs > 1:
            warnings.warn(f'epoch should be 1, but got {runner.max_epochs}. '
                          'profiler will slow down your training, so it is '
                          'not advised to set epoch greater than 1 if you use '
                          'ProfilerHook.')

        self.profiler = torch.profiler.profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=_on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops)

        self.profiler.__enter__()

    @master_only
    def after_train_iter(self, runner):
        self.profiler.step()

    @master_only
    def after_val_iter(self, runner):
        self.profiler.step()

    @master_only
    def after_run(self, runner):
        runner.logger.info('profiler may take a few minutes...')
        self.profiler.__exit__(None, None, None)
        if self.json_trace_path is not None:
            self.profiler.export_chrome_trace(self.json_trace_path)
