import random
import shutil

from tests.test_runner.test_hook.test_utils import \
    _build_demo_runner_without_hook

from mmcv.runner.hooks import HOOKS, Hook


def test_custom_hook():

    @HOOKS.register_module()
    class ToyHook(Hook):

        def __init__(self, info, *args, **kwargs):
            super().__init__()
            self.info = info

    runner = _build_demo_runner_without_hook('EpochBasedRunner', max_epochs=1)
    # test if custom_hooks is None
    runner.register_custom_hooks(None)
    assert len(runner.hooks) == 0
    # test if custom_hooks is dict list
    custom_hooks_cfg = [
        dict(type='ToyHook', priority=51, info=51),
        dict(type='ToyHook', priority=49, info=49)
    ]
    runner.register_custom_hooks(custom_hooks_cfg)
    assert [hook.info for hook in runner.hooks] == [49, 51]
    # test if custom_hooks is object and without priority
    runner.register_custom_hooks(ToyHook(info='default'))
    assert len(runner.hooks) == 3 and runner.hooks[1].info == 'default'
    shutil.rmtree(runner.work_dir)

    runner = _build_demo_runner_without_hook('EpochBasedRunner', max_epochs=1)
    # test custom_hooks with string priority setting
    priority_ranks = [
        'HIGHEST', 'VERY_HIGH', 'HIGH', 'ABOVE_NORMAL', 'NORMAL',
        'BELOW_NORMAL', 'LOW', 'VERY_LOW', 'LOWEST'
    ]
    random_priority_ranks = priority_ranks.copy()
    random.shuffle(random_priority_ranks)
    custom_hooks_cfg = [
        dict(type='ToyHook', priority=rank, info=rank)
        for rank in random_priority_ranks
    ]
    runner.register_custom_hooks(custom_hooks_cfg)
    assert [hook.info for hook in runner.hooks] == priority_ranks
    shutil.rmtree(runner.work_dir)

    runner = _build_demo_runner_without_hook('EpochBasedRunner', max_epochs=1)
    # test register_training_hooks order
    custom_hooks_cfg = [
        dict(type='ToyHook', priority=1, info='custom 1'),
        dict(type='ToyHook', priority='NORMAL', info='custom normal'),
        dict(type='ToyHook', priority=89, info='custom 89')
    ]
    runner.register_training_hooks(
        lr_config=ToyHook('lr'),
        optimizer_config=ToyHook('optimizer'),
        checkpoint_config=ToyHook('checkpoint'),
        log_config=dict(interval=1, hooks=[dict(type='ToyHook', info='log')]),
        momentum_config=ToyHook('momentum'),
        timer_config=ToyHook('timer'),
        custom_hooks_config=custom_hooks_cfg)
    # If custom hooks have same priority with default hooks, custom hooks
    # will be triggered after default hooks.
    hooks_order = [
        'custom 1', 'lr', 'momentum', 'optimizer', 'checkpoint',
        'custom normal', 'timer', 'custom 89', 'log'
    ]
    assert [hook.info for hook in runner.hooks] == hooks_order
    shutil.rmtree(runner.work_dir)
