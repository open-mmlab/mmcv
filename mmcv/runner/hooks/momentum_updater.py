from .hook import HOOKS, Hook
from .lr_updater import annealing_cos


class MomentumUpdaterHook(Hook):

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.9):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_momentum" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        self.base_momentum = []  # initial momentum for all param groups
        self.regular_momentum = [
        ]  # expected momentum if no warming up is performed

    def _set_momentum(self, runner, momentum_groups):
        for param_group, mom in zip(runner.optimizer.param_groups,
                                    momentum_groups):
            if 'momentum' in param_group.keys():
                param_group['momentum'] = mom
            elif 'betas' in param_group.keys():
                param_group['betas'] = (mom, param_group['betas'][1])

    def get_momentum(self, runner, base_momentum):
        raise NotImplementedError

    def get_regular_momentum(self, runner):
        return [
            self.get_momentum(runner, _base_momentum)
            for _base_momentum in self.base_momentum
        ]

    def get_warmup_momentum(self, cur_iters):
        if self.warmup == 'constant':
            warmup_momentum = [
                _momentum / self.warmup_ratio
                for _momentum in self.regular_momentum
            ]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_momentum = [
                _momentum / (1 - k) for _momentum in self.regular_mom
            ]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_momentum = [_momentum / k for _momentum in self.regular_mom]
        return warmup_momentum

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint,
        # if 'initial_momentum' is not saved,
        # it will be set according to the optimizer params
        for group in runner.optimizer.param_groups:
            if 'momentum' in group.keys():
                group.setdefault('initial_momentum', group['momentum'])
            else:
                group.setdefault('initial_momentum', group['betas'][0])
        self.base_momentum = [
            group['initial_momentum']
            for group in runner.optimizer.param_groups
        ]

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        self.regular_mom = self.get_regular_momentum(runner)
        self._set_momentum(runner, self.regular_mom)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_mom = self.get_regular_momentum(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_momentum(runner, self.regular_mom)
            else:
                warmup_momentum = self.get_warmup_momentum(cur_iter)
                self._set_momentum(runner, warmup_momentum)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_momentum(runner, self.regular_mom)
            else:
                warmup_momentum = self.get_warmup_momentum(cur_iter)
                self._set_momentum(runner, warmup_momentum)


@HOOKS.register_module()
class CosineAnnealingMomentumUpdaterHook(MomentumUpdaterHook):

    def __init__(self, min_momentum=None, min_momentum_ratio=None, **kwargs):
        assert (min_momentum is None) ^ (min_momentum_ratio is None)
        self.min_momentum = min_momentum
        self.min_momentum_ratio = min_momentum_ratio
        super(CosineAnnealingMomentumUpdaterHook, self).__init__(**kwargs)

    def get_momentum(self, runner, base_momentum):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        if self.min_momentum_ratio is not None:
            target_momentum = base_momentum * self.min_momentum_ratio
        else:
            target_momentum = self.min_momentum
        return annealing_cos(base_momentum, target_momentum,
                             progress / max_progress)


@HOOKS.register_module()
class CyclicMomentumUpdaterHook(MomentumUpdaterHook):
    """Cyclic momentum Scheduler.

    Implemet the cyclical momentum scheduler policy described in
    https://arxiv.org/pdf/1708.07120.pdf

    This momentum scheduler usually used together with the CyclicLRUpdater
    to improve the performance in the 3D detection area.

    Attributes:
        target_ratio (tuple[float]): Relative ratio of the lowest momentum and
            the highest momentum to the initial momentum.
        cyclic_times (int): Number of cycles during training
        step_ratio_up (float): The ratio of the increasing process of momentum
            in  the total cycle.
        by_epoch (bool): Whether to update momentum by epoch.
    """

    def __init__(self,
                 by_epoch=False,
                 target_ratio=(0.85 / 0.95, 1),
                 cyclic_times=1,
                 step_ratio_up=0.4,
                 **kwargs):
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio / 1e5)
        elif isinstance(target_ratio, tuple):
            target_ratio = (target_ratio[0], target_ratio[0] / 1e5) \
                if len(target_ratio) == 1 else target_ratio
        else:
            raise ValueError('target_ratio should be either float '
                             f'or tuple, got {type(target_ratio)}')

        assert len(target_ratio) == 2, \
            '"target_ratio" must be list or tuple of two floats'
        assert 0 <= step_ratio_up < 1.0, \
            '"step_ratio_up" must be in range [0,1)'

        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.momentum_phases = []  # init momentum_phases
        # currently only support by_epoch=False
        assert not by_epoch, \
            'currently only support "by_epoch" = False'
        super(CyclicMomentumUpdaterHook, self).__init__(by_epoch, **kwargs)

    def before_run(self, runner):
        super(CyclicMomentumUpdaterHook, self).before_run(runner)
        # initiate momentum_phases
        # total momentum_phases are separated as up and down
        max_iter_per_phase = runner.max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * max_iter_per_phase)
        self.momentum_phases.append(
            [0, iter_up_phase, max_iter_per_phase, 1, self.target_ratio[0]])
        self.momentum_phases.append([
            iter_up_phase, max_iter_per_phase, max_iter_per_phase,
            self.target_ratio[0], self.target_ratio[1]
        ])

    def get_momentum(self, runner, base_momentum):
        curr_iter = runner.iter
        for (start_iter, end_iter, max_iter_per_phase, start_ratio,
             end_ratio) in self.momentum_phases:
            curr_iter %= max_iter_per_phase
            if start_iter <= curr_iter < end_iter:
                progress = curr_iter - start_iter
                return annealing_cos(base_momentum * start_ratio,
                                     base_momentum * end_ratio,
                                     progress / (end_iter - start_iter))
