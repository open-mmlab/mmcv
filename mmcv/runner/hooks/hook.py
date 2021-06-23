# Copyright (c) Open-MMLab. All rights reserved.
from mmcv.utils import Registry

HOOKS = Registry('hook')


class Hook:

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def before_train_epoch(self, runner):
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        self.before_iter(runner)

    def before_val_iter(self, runner):
        self.before_iter(runner)

    def after_train_iter(self, runner):
        self.after_iter(runner)

    def after_val_iter(self, runner):
        self.after_iter(runner)

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        return runner.inner_iter + 1 == len(runner.data_loader)

    def is_last_epoch(self, runner):
        return runner.epoch + 1 == runner._max_epochs

    def is_last_iter(self, runner):
        return runner.iter + 1 == runner._max_iters

    def get_trigger_stages(self):
        stages = [
            'before_run', 'before_train_epoch', 'before_train_iter',
            'after_train_iter', 'after_train_epoch', 'before_val_epoch',
            'before_val_iter', 'after_val_iter', 'after_val_epoch', 'after_run'
        ]

        trigger_stages = set()
        for stage in stages:
            base_method = getattr(Hook, stage)
            sub_method = getattr(self.__class__, stage)
            if sub_method != base_method:
                trigger_stages.add(stage)

        # some methods will be triggered in multi stages
        # use this dict to map method to stages.
        method_stages_map = {
            'before_epoch': ['before_train_epoch', 'before_val_epoch'],
            'after_epoch': ['after_train_epoch', 'after_val_epoch'],
            'before_iter': ['before_train_iter', 'before_val_iter'],
            'after_iter': ['after_train_iter', 'after_val_iter'],
        }

        for method, map_stages in method_stages_map.items():
            base_method = getattr(Hook, method)
            sub_method = getattr(self.__class__, method)
            if sub_method != base_method:
                trigger_stages.update(map_stages)

        return [stage for stage in stages if stage in trigger_stages]
