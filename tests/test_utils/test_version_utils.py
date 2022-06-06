# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import pytest

from mmcv import get_git_hash, parse_version_info
from mmcv.utils import digit_version


def test_digit_version():
    assert digit_version('0.2.16') == (0, 2, 16, 0, 0, 0)
    assert digit_version('1.2.3') == (1, 2, 3, 0, 0, 0)
    assert digit_version('1.2.3rc0') == (1, 2, 3, 0, -1, 0)
    assert digit_version('1.2.3rc1') == (1, 2, 3, 0, -1, 1)
    assert digit_version('1.0rc0') == (1, 0, 0, 0, -1, 0)
    assert digit_version('1.0') == digit_version('1.0.0')
    assert digit_version('1.5.0+cuda90_cudnn7.6.3_lms') == digit_version('1.5')
    assert digit_version('1.0.0dev') < digit_version('1.0.0a')
    assert digit_version('1.0.0a') < digit_version('1.0.0a1')
    assert digit_version('1.0.0a') < digit_version('1.0.0b')
    assert digit_version('1.0.0b') < digit_version('1.0.0rc')
    assert digit_version('1.0.0rc1') < digit_version('1.0.0')
    assert digit_version('1.0.0') < digit_version('1.0.0post')
    assert digit_version('1.0.0post') < digit_version('1.0.0post1')
    assert digit_version('v1') == (1, 0, 0, 0, 0, 0)
    assert digit_version('v1.1.5') == (1, 1, 5, 0, 0, 0)
    with pytest.raises(AssertionError):
        digit_version('a')
    with pytest.raises(AssertionError):
        digit_version('1x')
    with pytest.raises(AssertionError):
        digit_version('1.x')


def test_parse_version_info():
    assert parse_version_info('0.2.16') == (0, 2, 16, 0, 0, 0)
    assert parse_version_info('1.2.3') == (1, 2, 3, 0, 0, 0)
    assert parse_version_info('1.2.3rc0') == (1, 2, 3, 0, 'rc', 0)
    assert parse_version_info('1.2.3rc1') == (1, 2, 3, 0, 'rc', 1)
    assert parse_version_info('1.0rc0') == (1, 0, 0, 0, 'rc', 0)


def _mock_cmd_success(cmd):
    return b'3b46d33e90c397869ad5103075838fdfc9812aa0'


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
