# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, Optional, Union

from mmcv.utils import scandir
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class NeptuneLoggerHook(LoggerHook):
    """Class to log metrics to NeptuneAI.

    It requires `Neptune`_ to be installed.

    Args:
        init_kwargs (dict): a dict contains the initialization keys as below:

            - project (str): Name of a project in the form project-name.
            - api_token (str): User’s API token. If None, the value of
              NEPTUNE_API_TOKEN environment variable will be taken.
              Set to neptune.ANONYMOUS_API_TOKEN to log metadata anonymously.
            - name (str, optional, default is 'Untitled'):
              Editable name of the run. Is displayed in the run information
              and can be added as a column in the runs table.
            - description (str): Editable description of the run.
              Is displayed in the run information and can be added
              as a column in the runs table.

            Check https://docs.neptune.ai/api/neptune/#init_run for more
            init arguments.
        interval (int): Logging interval (every k iterations). Default: 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than ``interval``. Default: True.
        log_artifact (bool): If True, artifacts in {work_dir} will be uploaded
            to neptune after training ends.
            Default: True
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be uploaded to neptune.
            Default: ('.log.json', '.log', '.py').
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: True.
        with_step (bool): If True, the step will be logged from
            ``self.get_iters``. Otherwise, step will not be logged.
            Default: True.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.

    .. _Neptune:
        https://docs.neptune.ai
    """

    def __init__(self,
                 init_kwargs: Optional[Dict] = None,
                 interval: int = 10,
                 log_artifact: bool = True,
                 out_suffix: Union[str, tuple] = ('.log.json', '.log', '.py'),
                 ignore_last: bool = True,
                 reset_flag: bool = True,
                 with_step: bool = True,
                 by_epoch: bool = True):

        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_neptune()
        self.init_kwargs = init_kwargs
        self.with_step = with_step
        self.log_artifact = log_artifact
        self.out_suffix = out_suffix

    def import_neptune(self) -> None:
        try:
            import neptune
        except ImportError:
            raise ImportError(
                'Please run "pip install -U neptune" to install neptune')
        self.neptune = neptune

    @master_only
    def before_run(self, runner) -> None:
        if self.init_kwargs:
            # neptune.init is deprecated.
            # Use neptune.init_run() instead.
            self.run = self.neptune.init_run(**self.init_kwargs)
        else:
            self.run = self.neptune.init_run()
        self.run.get_url()  # return a direct link to the run in Neptune

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        if tags:
            for tag_name, tag_value in tags.items():
                if self.with_step:
                    # As of neptune-client 0.16.14,
                    # append() and extend() are the preferred methods.
                    self.run[tag_name].append(
                        tag_value, step=self.get_iter(runner))
                else:
                    tags['global_step'] = self.get_iter(runner)
                    self.run[tag_name].append(tags)

    @master_only
    def after_run(self, runner) -> None:
        if self.log_artifact:
            for filename in scandir(runner.work_dir, self.out_suffix, True):
                local_filepath = osp.join(runner.work_dir, filename)
                self.run[filename].track_files(local_filepath)

        self.run.stop()
