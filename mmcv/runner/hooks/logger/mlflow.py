# Copyright (c) Open-MMLab. All rights reserved.
import numbers

from mmcv.runner import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module
class MlflowLoggerHook(LoggerHook):

    def __init__(self,
                 experiment_name,
                 tracking_uri = None,
                 tags = None,
                 log_model=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(MlflowLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag)
        self.import_mlflow()
        self._mlflow_client = self.mlflow.tracking.MlflowClient(tracking_uri)
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
        experiment = self._mlflow_client.get_experiment_by_name(self.experiment_name)
        if experiment:
            self._experiment_id = experiment.experiment_id
        else:
            runner.logger.warning(f'Experiment with name {self.experiment_name} not found. Creating it.')
            self._experiment_id = self._mlflow_client.create_experiment(name=self.experiment_name)

        if self.mlflow.active_run():
            run = self._mlflow_client.get_run(self.mlflow.active_run())
        else:
            run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=self.tags)
        
        self._run_id = run.info.run_id

    @master_only
    def log(self, runner):
        for var, val in runner.log_buffer.output.items():
            if var in ['time', 'data_time']:
                continue
            tag = '{}/{}'.format(var, runner.mode)
            self._mlflow_client.log_metric(self._run_id, tag, val, step=runner.iter)

    @master_only
    def after_run(self, runner):
        if self.log_model:
            self.mlflow_pytorch.log_model(runner.model, "models")
        self._mlflow_client.set_terminated(self._run_id)
