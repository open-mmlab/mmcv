import pytest

from mmcv.engine import default_args_parser
from mmcv.utils import Config


def test_default_args():
    parser = default_args_parser()
    known_args, cfg_args = parser.parse_known_args(
        './configs/config.py dist_params.port 2999'.split())
    assert cfg_args == ['dist_params.port', '2999']

    cfg = Config(dict(a=1, b=dict(b1=[0, 1])))

    # test ValueError when strict=False and argument does not start
    # with "--"
    with pytest.raises(ValueError):
        cfg.merge_from_arg_list(cfg_args)

    cfg.merge_from_arg_list(cfg_args, strict=False)
    assert cfg.dist_params.port == 2999

    # test nargs before unknow args
    known_args, cfg_args = parser.parse_known_args(
        './configs/config.py --gpu-ids 1 2 3 --work-dir work_dirs'.split())
    assert cfg_args == ['--work-dir', 'work_dirs']

    cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
    cfg.merge_from_arg_list(cfg_args)
    assert cfg.work_dir == 'work_dirs'

    # test float, bool type and list
    known_args, cfg_args = parser.parse_known_args(
        './configs/config.py --a-b a_b --list 1,2,3.5 --bool True'.split())
    assert cfg_args == ['--a-b', 'a_b', '--list', '1,2,3.5', '--bool', 'True']

    cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
    cfg.merge_from_arg_list(cfg_args)
    assert cfg.a_b == 'a_b'
    assert cfg.list == [1, 2, 3.5]
    assert cfg.bool
