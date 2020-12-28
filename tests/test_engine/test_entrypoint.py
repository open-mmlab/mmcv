import os.path as osp
import tempfile
import time

from mmcv.engine import default_args_parser, gather_info, setup_cfg, setup_envs
from mmcv.utils import collect_env, get_logger

data_path = osp.join(osp.dirname(osp.dirname(__file__)), 'data')


def test_entrypoint():
    with tempfile.TemporaryDirectory() as tmp_dir:
        opts = f'{data_path}/config/a.py --work-dir {tmp_dir} --log-level INFO'
        args, cfg_opts = default_args_parser().parse_known_args(opts.split())
        cfg = setup_cfg(args, cfg_opts)

        setup_envs(cfg)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_logger(
            name='mmcv', log_file=log_file, log_level=cfg.log_level)

        meta = gather_info(cfg, logger, collect_env())
        assert meta is not None
