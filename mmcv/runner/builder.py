from ..utils import Registry, build_from_cfg
import copy

RUNNERS = Registry('runner')
RUNNER_BUILDERS = Registry('runner builder')


def build_runner_constructor(cfg):
    return build_from_cfg(cfg, RUNNER_BUILDERS)


def build_runner(cfg, default_args=None):
    runner_cfg = copy.deepcopy(cfg)
    constructor_type = runner_cfg.pop(
        'constructor', 'DefaultRunnerConstructor')
    runner_constructor = build_runner_constructor(
        dict(
            type=constructor_type,
            runner_cfg=runner_cfg,
            default_args=default_args))
    runner = runner_constructor()
    return runner
