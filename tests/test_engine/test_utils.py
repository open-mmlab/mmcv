from mmcv.engine import default_args_parser


def test_default_args():
    parser = default_args_parser()
    known_args, cfg_args = parser.parse_known_args(
        ['./configs/config.py', 'dist_params.port', '2999'])
    assert cfg_args == ['dist_params.port', '2999']
