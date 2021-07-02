from ..utils import Registry, build_from_cfg

RUNNERS = Registry('runner')


def build_runner(cfg, default_args=None):
    return build_from_cfg(cfg, RUNNERS, default_args=default_args)
