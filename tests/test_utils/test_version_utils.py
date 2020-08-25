from unittest.mock import patch

from mmcv import digit_version, get_git_hash, parse_version_info


def test_digit_version():
    assert digit_version('0.2.16') == (0, 2, 16)
    assert digit_version('1.2.3') == (1, 2, 3)
    assert digit_version('1.2.3rc0') == (1, 2, 2, 0)
    assert digit_version('1.2.3rc1') == (1, 2, 2, 1)
    assert digit_version('1.0rc0') == (1, -1, 0)


def test_parse_version_info():
    assert parse_version_info('0.2.16') == (0, 2, 16)
    assert parse_version_info('1.2.3') == (1, 2, 3)
    assert parse_version_info('1.2.3rc0') == (1, 2, 3, 'rc0')
    assert parse_version_info('1.2.3rc1') == (1, 2, 3, 'rc1')
    assert parse_version_info('1.0rc0') == (1, 0, 'rc0')


def _mock_cmd_success(cmd):
    return '3b46d33e90c397869ad5103075838fdfc9812aa0'.encode('ascii')


def _mock_cmd_fail(cmd):
    raise OSError


def test_get_git_hash():
    with patch('mmcv.utils.version_utils._minimal_ext_cmd', _mock_cmd_success):
        assert get_git_hash() == '3b46d33e90c397869ad5103075838fdfc9812aa0'
        assert get_git_hash(digits=6) == '3b46d3'
        assert get_git_hash(digits=100) == get_git_hash()
    with patch('mmcv.utils.version_utils._minimal_ext_cmd', _mock_cmd_fail):
        assert get_git_hash() == 'unknown'
        assert get_git_hash(fallback='n/a') == 'n/a'
