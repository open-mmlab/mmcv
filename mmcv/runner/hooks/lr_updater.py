from __future__ import division

from math import cos, pi, floor

from .hook import Hook


class LrUpdaterHook(Hook):

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 **kwargs):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    '"{}" is not a supported type for warming up, valid types'
                    ' are "constant" and "linear"'.format(warmup))
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        for param_group, lr in zip(runner.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, runner):
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in runner.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in runner.optimizer.param_groups
        ]

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


class FixedLrUpdaterHook(LrUpdaterHook):

    def __init__(self, **kwargs):
        super(FixedLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        return base_lr


class StepLrUpdaterHook(LrUpdaterHook):

    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(StepLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        if isinstance(self.step, int):
            return base_lr * (self.gamma**(progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_lr * self.gamma**exp


class ExpLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.gamma**progress


class PolyLrUpdaterHook(LrUpdaterHook):

    def __init__(self, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        super(PolyLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


class InvLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * (1 + self.gamma * progress)**(-self.power)


class CosineLrUpdaterHook(LrUpdaterHook):

    def __init__(self, target_lr=0, **kwargs):
        self.target_lr = target_lr
        super(CosineLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        return self.target_lr + 0.5 * (base_lr - self.target_lr) * \
            (1 + cos(pi * (progress / max_progress)))


class CyclicLrUpdaterHook(LrUpdaterHook):

    def __init__(self,
                 max_lr,
                 final_cycle_lr=None,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 cycle_momentum=True,
                 max_momentum=0.9,
                 scale_fn=None,
                 scale_mode='cycle',
                 debug=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert self.by_epoch is False, \
            '"by_epoch" must be False.'

        self.max_lr = max_lr
        self.final_cycle_lr = final_cycle_lr
        self.gamma = gamma
        self.cycle_momentum = cycle_momentum is True
        self.max_momentum = max_momentum

        step_size_up = float(step_size_up)
        step_size_down = float(
            step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        self.mode = mode
        modes = ('triangular', 'triangular2', 'exp_range')
        if (mode not in modes) and (scale_fn is None):
            raise ValueError('mode {} is invalid and scale_fn is None.\n'
                             ' Valid modes are {}'.format(mode, modes))
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.debug = debug
        self._init_2_needed = True

    def _init_2(self, runner):
        if self.by_epoch:
            self.max_progress = runner.max_epochs
        else:
            self.max_progress = runner.max_iters
        self.total_cycle = floor(1 + self.max_progress / self.total_size)
        self._init_2_needed = False

    @staticmethod
    def _triangular_scale_fn(_):
        return 1.

    @staticmethod
    def _triangular2_scale_fn(x):
        return 1 / (2.**(x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**x

    def get_lr(self, runner, base_lr):

        if self._init_2_needed:
            self._init_2(runner)

        progress = runner.epoch if self.by_epoch else runner.iter
        cycle = floor(1 + progress / self.total_size)

        if (self.final_cycle_lr is not None) and (self.total_cycle > 1) and (
                cycle >= self.total_cycle):
            if self.debug:
                print('1-CyclicLrUpdaterHook, iter:{},lr:{}'.format(
                    progress, self.final_cycle_lr))
            return self.final_cycle_lr

        x = 1. + progress / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        base_height = (self.max_lr - base_lr) * scale_factor

        if self.scale_mode == 'cycle':
            lr = base_lr + base_height * self.scale_fn(cycle)
        else:
            lr = base_lr + base_height * self.scale_fn(progress)

        if self.cycle_momentum:
            base_height = (
                self.max_momentum -
                runner.optimizer.defaults['momentum']) \
                * scale_factor
            if self.scale_mode == 'cycle':
                momentum = self.max_momentum - \
                    base_height * self.scale_fn(cycle)
            else:
                momentum = self.max_momentum - base_height * self.scale_fn(
                    progress)
            for param_group in runner.optimizer.param_groups:
                param_group['momentum'] = momentum
            if self.debug:
                print('CyclicLrUpdaterHook,iter:{},lr:{},momentum:{}'.format(
                    progress, lr, momentum))
        return lr
