import os
import os.path as osp
import warnings
from math import inf

import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from .hook import Hook


class EvalHook(Hook):
    """Non-Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader, whose dataset has
            implemented ``evaluate`` function.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval. Default: 1.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            default: True.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be save in ``runner.meta['hook_msgs']``.
            Options are the evaluation metrics to the test dataset. e.g.,
            ``bbox_mAP``, ``segm_mAP`` for bbox detection and instance
            segmentation. ``AR@100`` for proposal recall. If ``save_best`` is
            ``auto``, the first key of the returned ``OrderedDict`` result
            will be used. The interval of ``EvalHook`` should be
            divisible of that in ``CheckpointHook``. Default: None.
        rule (str | None, optional): Comparison rule for best score. If set to
            None, it will infer a reasonable rule. Keys such as 'acc', 'top'
            .etc will be inferred by 'greater' rule. Keys contain 'loss' will
            be inferred by 'less' rule. Options are 'greater', 'less', None.
            Default: None.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.

    Notes:
        If new arguments are added for EvalHook, tools/test.py,
        tools/eval_metric.py may be affected.
    """

    # Since the key for determine greater or less is related to the downstream
    # tasks, downstream repos may need to overwrite the following inner
    # variable accordingly.

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    greater_keys = ['acc', 'top', 'AR@', 'auc', 'precision', 'mAP']
    less_keys = ['loss']

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f'dataloader must be a pytorch DataLoader, '
                            f'but got {type(dataloader)}')

        if interval <= 0:
            raise ValueError(f'interval must be a positive number, '
                             f'but got {interval}')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean'

        if start is not None and start < 0:
            raise ValueError(f'The evaluation start epoch {start} is smaller '
                             f'than 0')

        self.dataloader = dataloader
        self.interval = interval
        self.start = start
        self.by_epoch = by_epoch

        assert isinstance(save_best, str) or save_best is None, \
            '""save_best"" should be a str or None ' \
            f'rather than {type(save_best)}'
        self.save_best = save_best
        self.eval_kwargs = eval_kwargs
        self.initial_flag = True

        if self.save_best is not None:
            self.best_ckpt_path = None
            self._init_rule(rule, self.save_best)

    def _init_rule(self, rule, key_indicator):
        """Initialize rule, key_indicator, comparison_func, and best score.

        Here is the rule to determine which rule is used for key indicator
        when the rule is not specific:
        1. If the key indicator is in ``self.greater_keys``, the rule will be
           specified as 'greater'.
        2. Or if the key indicator is in ``self.less_keys``, the rule will be
           specified as 'less'.
        3. Or if the key indicator is equal to the substring in any one item
           in ``self.greater_keys``, the rule will be specified as 'greater'.
        4. Or if the key indicator is equal to the substring in any one item
           in ``self.less_keys``, the rule will be specified as 'less'.

        Args:
            rule (str | None): Comparison rule for best score.
            key_indicator (str | None): Key indicator to determine the
                comparison rule.
        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')

        if rule is None:
            if key_indicator != 'auto':
                if key_indicator in self.greater_keys:
                    rule = 'greater'
                elif key_indicator in self.less_keys:
                    rule = 'less'
                elif any(key in key_indicator for key in self.greater_keys):
                    rule = 'greater'
                elif any(key in key_indicator for key in self.less_keys):
                    rule = 'less'
                else:
                    raise ValueError(f'Cannot infer the rule for key '
                                     f'{key_indicator}, thus a specific rule '
                                     f'must be specified.')
        self.rule = rule
        self.key_indicator = key_indicator
        if self.rule is not None:
            self.compare_func = self.rule_map[self.rule]

    def before_run(self, runner):
        if self.save_best is not None:
            if runner.meta is None:
                warnings.warn('runner.meta is None. Creating an empty one.')
                runner.meta = dict()
            runner.meta.setdefault('hook_msgs', dict())

    def before_train_iter(self, runner):
        """Evaluate the model only at the start of training by iteration."""
        if self.by_epoch or not self.initial_flag:
            return
        if self.start is not None and runner.iter >= self.start:
            self.after_train_iter(runner)
        self.initial_flag = False

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        if not (self.by_epoch and self.initial_flag):
            return
        if self.start is not None and runner.epoch >= self.start:
            self.after_train_epoch(runner)
        self.initial_flag = False

    def after_train_iter(self, runner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch:
            self._do_evaluate(runner)

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        if self.by_epoch:
            self._do_evaluate(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmcv.engine import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)

    def _should_evaluate(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7...
            # if start==3 and interval==2
            if (current + 1 - self.start) % self.interval:
                return False
        return True

    def _save_ckpt(self, runner, key_score):
        """Save the best checkpoint.

        It will compare the score according to the compare function, write
        related information (best score, best checkpoint path) and save the
        best checkpoint into ``work_dir``.
        """
        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        best_score = runner.meta['hook_msgs'].get(
            'best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            runner.meta['hook_msgs']['best_score'] = best_score

            if self.best_ckpt_path and osp.isfile(self.best_ckpt_path):
                os.remove(self.best_ckpt_path)

            best_ckpt_name = f'best_{self.key_indicator}_{current}.pth'
            runner.save_checkpoint(
                runner.work_dir, best_ckpt_name, create_symlink=False)
            self.best_ckpt_path = osp.join(runner.work_dir, best_ckpt_name)
            runner.meta['hook_msgs']['best_ckpt'] = self.best_ckpt_path
            runner.logger.info(
                f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(
                f'Best {self.key_indicator} is {best_score:0.4f} '
                f'at {cur_time} {cur_type}.')

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader, whose dataset has
            implemented ``evaluate`` function.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval. Default: 1.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            default: True.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be save in ``runner.meta['hook_msgs']``.
            Options are the evaluation metrics to the test dataset. e.g.,
            ``bbox_mAP``, ``segm_mAP`` for bbox detection and instance
            segmentation. ``AR@100`` for proposal recall. If ``save_best`` is
            ``auto``, the first key of the returned ``OrderedDict`` result
            will be used. The interval of ``EvalHook`` should depend on
            ``CheckpointHook``. Default: None.
        rule (str | None, optional): Comparison rule for best score. If set to
            None, it will infer a reasonable rule. Keys such as 'acc', 'top'
            .etc will be inferred by 'greater' rule. Keys contain 'loss' will
            be inferred by 'less' rule. Options are 'greater', 'less', None.
            Default: None.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        broadcast_bn_buffer (bool): Whether to broadcast the
            buffer(running_mean and running_var) of rank 0 to other rank
            before evaluation. Default: True.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 broadcast_bn_buffer=True,
                 tmpdir=None,
                 gpu_collect=False,
                 **eval_kwargs):
        super().__init__(
            dataloader,
            start=start,
            interval=interval,
            by_epoch=by_epoch,
            save_best=save_best,
            rule=rule,
            **eval_kwargs)
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmcv.engine import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
