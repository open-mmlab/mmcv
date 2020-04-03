# Copyright (c) Open-MMLab. All rights reserved.
import numbers

from mmcv.runner import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module
class MlflowLoggerHook(LoggerHook):

    def __init__(self,
                 experiment_name=None,
                 tags = None,
                 log_model=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(MlflowLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag)
        self.import_mlflow()
        self.experiment_name = experiment_name
        self.tags = tags
        self.log_model = log_model

    def import_mlflow(self):
        try:
            import mlflow
            import mlflow.pytorch as mlflow_pytorch
        except ImportError:
            raise ImportError(
                'Please run "pip install mlflow" to install mlflow')
        self.mlflow = mlflow
        self.mlflow_pytorch = mlflow_pytorch

    @master_only
    def before_run(self, runner):
        if self.experiment_name is not None:
            self.mlflow.set_experiment(experiment_name)

    @master_only
    def log(self, runner):
        metrics = {}
        for var, val in runner.log_buffer.output.items():
            if var in ['time', 'data_time']:
                continue
            tag = '{}/{}'.format(var, runner.mode)
            metrics[tag] = val
        self.mlflow.log_metrics(metrics, step=runner.iter)

    @master_only
    def after_run(self, runner):
        if self.log_model:
            self.mlflow_pytorch.log_model(runner.model, "models")
