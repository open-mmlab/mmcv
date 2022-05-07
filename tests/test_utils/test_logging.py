# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import platform
import tempfile
from unittest.mock import patch

import pytest

from mmcv import get_logger, print_log

if platform.system() == 'Windows':
    import regex as re
else:
    import re


@patch('torch.distributed.get_rank', lambda: 0)
@patch('torch.distributed.is_initialized', lambda: True)
@patch('torch.distributed.is_available', lambda: True)
def test_get_logger_rank0():
    logger = get_logger('rank0.pkg1')
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == logging.INFO

    logger = get_logger('rank0.pkg2', log_level=logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert logger.handlers[0].level == logging.DEBUG

    # the name can not be used to open the file a second time in windows,
    # so `delete` should be set as `False` and we need to manually remove it
    # more details can be found at https://github.com/open-mmlab/mmcv/pull/1077
    with tempfile.NamedTemporaryFile(delete=False) as f:
        logger = get_logger('rank0.pkg3', log_file=f.name)
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 2
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert isinstance(logger.handlers[1], logging.FileHandler)
        logger_pkg3 = get_logger('rank0.pkg3')
        assert id(logger_pkg3) == id(logger)
        # flushing and closing all handlers in order to remove `f.name`
        logging.shutdown()

    os.remove(f.name)

    logger_pkg3 = get_logger('rank0.pkg3.subpkg')
    assert logger_pkg3.handlers == logger_pkg3.handlers


@patch('torch.distributed.get_rank', lambda: 1)
@patch('torch.distributed.is_initialized', lambda: True)
@patch('torch.distributed.is_available', lambda: True)
def test_get_logger_rank1():
    logger = get_logger('rank1.pkg1')
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == logging.INFO

    # the name can not be used to open the file a second time in windows,
    # so `delete` should be set as `False` and we need to manually remove it
    # more details can be found at https://github.com/open-mmlab/mmcv/pull/1077
    with tempfile.NamedTemporaryFile(delete=False) as f:
        logger = get_logger('rank1.pkg2', log_file=f.name)
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 1
        assert logger.handlers[0].level == logging.INFO
        # flushing and closing all handlers in order to remove `f.name`
        logging.shutdown()

    os.remove(f.name)


def test_print_log_print(capsys):
    print_log('welcome', logger=None)
    out, _ = capsys.readouterr()
    assert out == 'welcome\n'


def test_print_log_silent(capsys, caplog):
    print_log('welcome', logger='silent')
    out, _ = capsys.readouterr()
    assert out == ''
    assert len(caplog.records) == 0


def test_print_log_logger(caplog):
    print_log('welcome', logger='mmcv')
    assert caplog.record_tuples[-1] == ('mmcv', logging.INFO, 'welcome')

    print_log('welcome', logger='mmcv', level=logging.ERROR)
    assert caplog.record_tuples[-1] == ('mmcv', logging.ERROR, 'welcome')

    # the name can not be used to open the file a second time in windows,
    # so `delete` should be set as `False` and we need to manually remove it
    # more details can be found at https://github.com/open-mmlab/mmcv/pull/1077
    with tempfile.NamedTemporaryFile(delete=False) as f:
        logger = get_logger('abc', log_file=f.name)
        print_log('welcome', logger=logger)
        assert caplog.record_tuples[-1] == ('abc', logging.INFO, 'welcome')
        with open(f.name) as fin:
            log_text = fin.read()
            regex_time = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'
            match = re.fullmatch(regex_time + r' - abc - INFO - welcome\n',
                                 log_text)
            assert match is not None
        # flushing and closing all handlers in order to remove `f.name`
        logging.shutdown()

    os.remove(f.name)


def test_print_log_exception():
    with pytest.raises(TypeError):
        print_log('welcome', logger=0)
