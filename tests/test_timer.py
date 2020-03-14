# Copyright (c) Open-MMLab. All rights reserved.
import time

import pytest

import mmcv


def test_timer_init():
    timer = mmcv.Timer(start=False)
    assert not timer.is_running
    timer.start()
    assert timer.is_running
    timer = mmcv.Timer()
    assert timer.is_running


def test_timer_run():
    timer = mmcv.Timer()
    time.sleep(1)
    assert abs(timer.since_start() - 1) < 1e-2
    time.sleep(1)
    assert abs(timer.since_last_check() - 1) < 1e-2
    assert abs(timer.since_start() - 2) < 1e-2
    timer = mmcv.Timer(False)
    with pytest.raises(mmcv.TimerError):
        timer.since_start()
    with pytest.raises(mmcv.TimerError):
        timer.since_last_check()


def test_timer_context(capsys):
    with mmcv.Timer():
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert abs(float(out) - 1) < 1e-2
    with mmcv.Timer(print_tmpl='time: {:.1f}s'):
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert out == 'time: 1.0s\n'
