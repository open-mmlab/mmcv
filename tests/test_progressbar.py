import sys
import time
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import pytest

import mmcv


def reset_string_io(io):
    io.truncate(0)
    io.seek(0)


if sys.version_info[0] == 2:
    pytest.skip('skipping tests for python 2', allow_module_level=True)


class TestProgressBar(object):

    def test_start(self):
        out = StringIO()
        bar_width = 20
        # without total task num
        prog_bar = mmcv.ProgressBar(bar_width=bar_width, file=out)
        assert out.getvalue() == 'completed: 0, elapsed: 0s'
        reset_string_io(out)
        prog_bar = mmcv.ProgressBar(bar_width=bar_width, start=False, file=out)
        assert out.getvalue() == ''
        reset_string_io(out)
        prog_bar.start()
        assert out.getvalue() == 'completed: 0, elapsed: 0s'
        # with total task num
        reset_string_io(out)
        prog_bar = mmcv.ProgressBar(10, bar_width=bar_width, file=out)
        assert out.getvalue() == '[{}] 0/10, elapsed: 0s, ETA:'.format(
            ' ' * bar_width)
        reset_string_io(out)
        prog_bar = mmcv.ProgressBar(
            10, bar_width=bar_width, start=False, file=out)
        assert out.getvalue() == ''
        reset_string_io(out)
        prog_bar.start()
        assert out.getvalue() == '[{}] 0/10, elapsed: 0s, ETA:'.format(
            ' ' * bar_width)

    def test_update(self):
        out = StringIO()
        bar_width = 20
        # without total task num
        prog_bar = mmcv.ProgressBar(bar_width=bar_width, file=out)
        time.sleep(1)
        reset_string_io(out)
        prog_bar.update()
        assert out.getvalue() == 'completed: 1, elapsed: 1s, 1.0 tasks/s'
        reset_string_io(out)
        # with total task num
        prog_bar = mmcv.ProgressBar(10, bar_width=bar_width, file=out)
        time.sleep(1)
        reset_string_io(out)
        prog_bar.update()
        assert out.getvalue() == ('\r[{}] 1/10, 1.0 task/s, '
                                  'elapsed: 1s, ETA:     9s'.format('>' * 2 +
                                                                    ' ' * 18))


def sleep_1s(num):
    time.sleep(1)
    return num


def test_track_progress_list():
    out = StringIO()
    ret = mmcv.track_progress(sleep_1s, [1, 2, 3], bar_width=3, file=out)
    assert out.getvalue() == (
        '[   ] 0/3, elapsed: 0s, ETA:'
        '\r[>  ] 1/3, 1.0 task/s, elapsed: 1s, ETA:     2s'
        '\r[>> ] 2/3, 1.0 task/s, elapsed: 2s, ETA:     1s'
        '\r[>>>] 3/3, 1.0 task/s, elapsed: 3s, ETA:     0s\n')
    assert ret == [1, 2, 3]


def test_track_progress_iterator():
    out = StringIO()
    ret = mmcv.track_progress(
        sleep_1s, ((i for i in [1, 2, 3]), 3), bar_width=3, file=out)
    assert out.getvalue() == (
        '[   ] 0/3, elapsed: 0s, ETA:'
        '\r[>  ] 1/3, 1.0 task/s, elapsed: 1s, ETA:     2s'
        '\r[>> ] 2/3, 1.0 task/s, elapsed: 2s, ETA:     1s'
        '\r[>>>] 3/3, 1.0 task/s, elapsed: 3s, ETA:     0s\n')
    assert ret == [1, 2, 3]


def test_track_iter_progress():
    out = StringIO()
    ret = []
    for num in mmcv.track_iter_progress([1, 2, 3], bar_width=3, file=out):
        ret.append(sleep_1s(num))
    assert out.getvalue() == (
        '[   ] 0/3, elapsed: 0s, ETA:'
        '\r[>  ] 1/3, 1.0 task/s, elapsed: 1s, ETA:     2s'
        '\r[>> ] 2/3, 1.0 task/s, elapsed: 2s, ETA:     1s'
        '\r[>>>] 3/3, 1.0 task/s, elapsed: 3s, ETA:     0s\n')
    assert ret == [1, 2, 3]


def test_track_enum_progress():
    out = StringIO()
    ret = []
    count = []
    for i, num in enumerate(
            mmcv.track_iter_progress([1, 2, 3], bar_width=3, file=out)):
        ret.append(sleep_1s(num))
        count.append(i)
    assert out.getvalue() == (
        '[   ] 0/3, elapsed: 0s, ETA:'
        '\r[>  ] 1/3, 1.0 task/s, elapsed: 1s, ETA:     2s'
        '\r[>> ] 2/3, 1.0 task/s, elapsed: 2s, ETA:     1s'
        '\r[>>>] 3/3, 1.0 task/s, elapsed: 3s, ETA:     0s\n')
    assert ret == [1, 2, 3]
    assert count == [0, 1, 2]


def test_track_parallel_progress_list():
    out = StringIO()
    results = mmcv.track_parallel_progress(
        sleep_1s, [1, 2, 3, 4], 2, bar_width=4, file=out)
    assert out.getvalue() == (
        '[    ] 0/4, elapsed: 0s, ETA:'
        '\r[>   ] 1/4, 1.0 task/s, elapsed: 1s, ETA:     3s'
        '\r[>>  ] 2/4, 2.0 task/s, elapsed: 1s, ETA:     1s'
        '\r[>>> ] 3/4, 1.5 task/s, elapsed: 2s, ETA:     1s'
        '\r[>>>>] 4/4, 2.0 task/s, elapsed: 2s, ETA:     0s\n')
    assert results == [1, 2, 3, 4]


def test_track_parallel_progress_iterator():
    out = StringIO()
    results = mmcv.track_parallel_progress(
        sleep_1s, ((i for i in [1, 2, 3, 4]), 4), 2, bar_width=4, file=out)
    assert out.getvalue() == (
        '[    ] 0/4, elapsed: 0s, ETA:'
        '\r[>   ] 1/4, 1.0 task/s, elapsed: 1s, ETA:     3s'
        '\r[>>  ] 2/4, 2.0 task/s, elapsed: 1s, ETA:     1s'
        '\r[>>> ] 3/4, 1.5 task/s, elapsed: 2s, ETA:     1s'
        '\r[>>>>] 4/4, 2.0 task/s, elapsed: 2s, ETA:     0s\n')
    assert results == [1, 2, 3, 4]
