# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import sys
import tempfile
import unittest.mock as mock
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from mmcv.fileio.file_client import PetrelBackend
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EpochBasedRunner
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.runner import IterBasedRunner
from mmcv.utils import get_logger, scandir

sys.modules['petrel_client'] = MagicMock()
sys.modules['petrel_client.client'] = MagicMock()


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0
        self.eval_result = [1, 4, 3, 7, 2, -3, 4, 6]

    def __getitem__(self, idx):
        results = dict(x=torch.tensor([1]))
        return results

    def __len__(self):
        return 1

    @mock.create_autospec
    def evaluate(self, results, logger=None):
        pass


class EvalDataset(ExampleDataset):

    def evaluate(self, results, logger=None):
        acc = self.eval_result[self.index]
        output = OrderedDict(
            acc=acc, index=self.index, score=acc, loss_top=acc)
        self.index += 1
        return output


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x, **kwargs):
        return self.param * x

    def train_step(self, data_batch, optimizer, **kwargs):
        return {'loss': torch.sum(self(data_batch['x']))}

    def val_step(self, data_batch, optimizer, **kwargs):
        return {'loss': torch.sum(self(data_batch['x']))}


def _build_epoch_runner():
    model = Model()
    tmp_dir = tempfile.mkdtemp()

    runner = EpochBasedRunner(
        model=model, work_dir=tmp_dir, logger=get_logger('demo'))
    return runner


def _build_iter_runner():
    model = Model()
    tmp_dir = tempfile.mkdtemp()

    runner = IterBasedRunner(
        model=model, work_dir=tmp_dir, logger=get_logger('demo'))
    return runner


class EvalHook(BaseEvalHook):

    _default_greater_keys = ['acc', 'top']
    _default_less_keys = ['loss', 'loss_top']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DistEvalHook(BaseDistEvalHook):

    greater_keys = ['acc', 'top']
    less_keys = ['loss', 'loss_top']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def test_eval_hook():
    with pytest.raises(AssertionError):
        # `save_best` should be a str
        test_dataset = Model()
        data_loader = DataLoader(test_dataset)
        EvalHook(data_loader, save_best=True)

    with pytest.raises(TypeError):
        # dataloader must be a pytorch DataLoader
        test_dataset = Model()
        data_loader = [DataLoader(test_dataset)]
        EvalHook(data_loader)

    with pytest.raises(ValueError):
        # key_indicator must be valid when rule_map is None
        test_dataset = ExampleDataset()
        data_loader = DataLoader(test_dataset)
        EvalHook(data_loader, save_best='unsupport')

    with pytest.raises(KeyError):
        # rule must be in keys of rule_map
        test_dataset = ExampleDataset()
        data_loader = DataLoader(test_dataset)
        EvalHook(data_loader, save_best='auto', rule='unsupport')

    # if eval_res is an empty dict, print a warning information
    with pytest.warns(UserWarning) as record_warnings:

        class _EvalDataset(ExampleDataset):

            def evaluate(self, results, logger=None):
                return {}

        test_dataset = _EvalDataset()
        data_loader = DataLoader(test_dataset)
        eval_hook = EvalHook(data_loader, save_best='auto')
        runner = _build_epoch_runner()
        runner.register_hook(eval_hook)
        runner.run([data_loader], [('train', 1)], 1)
    # Since there will be many warnings thrown, we just need to check if the
    # expected exceptions are thrown
    expected_message = ('Since `eval_res` is an empty dict, the behavior to '
                        'save the best checkpoint will be skipped in this '
                        'evaluation.')
    for warning in record_warnings:
        if str(warning.message) == expected_message:
            break
    else:
        assert False

    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset)
    model = Model()
    data_loader = DataLoader(test_dataset)
    eval_hook = EvalHook(data_loader, save_best=None)

    with tempfile.TemporaryDirectory() as tmpdir:

        # total_epochs = 1
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)
        test_dataset.evaluate.assert_called_with(
            test_dataset, [torch.tensor([1])], logger=runner.logger)
        assert runner.meta is None or 'best_score' not in runner.meta[
            'hook_msgs']
        assert runner.meta is None or 'best_ckpt' not in runner.meta[
            'hook_msgs']

    # when `save_best` is set to 'auto', first metric will be used.
    loader = DataLoader(EvalDataset())
    model = Model()
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(data_loader, interval=1, save_best='auto')

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_acc_epoch_4.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == ckpt_path
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 7

    # total_epochs = 8, return the best acc and corresponding epoch
    loader = DataLoader(EvalDataset())
    model = Model()
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(data_loader, interval=1, save_best='acc')

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_acc_epoch_4.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == ckpt_path
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 7

    # total_epochs = 8, return the best loss_top and corresponding epoch
    loader = DataLoader(EvalDataset())
    model = Model()
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(data_loader, interval=1, save_best='loss_top')

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_loss_top_epoch_6.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == ckpt_path
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == -3

    # total_epochs = 8, return the best score and corresponding epoch
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(
        data_loader, interval=1, save_best='score', rule='greater')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_score_epoch_4.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == ckpt_path
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 7

    # total_epochs = 8, return the best score using less compare func
    # and indicate corresponding epoch
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(data_loader, save_best='acc', rule='less')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_acc_epoch_6.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == ckpt_path
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == -3

    # Test the EvalHook when resume happened
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(data_loader, save_best='acc')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 2)

        old_ckpt_path = osp.join(tmpdir, 'best_acc_epoch_2.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == old_ckpt_path
        assert osp.exists(old_ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 4

        resume_from = old_ckpt_path
        loader = DataLoader(ExampleDataset())
        eval_hook = EvalHook(data_loader, save_best='acc')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)

        runner.resume(resume_from)
        assert runner.meta['hook_msgs']['best_ckpt'] == old_ckpt_path
        assert osp.exists(old_ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 4

        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_acc_epoch_4.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == ckpt_path
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == 7
        assert not osp.exists(old_ckpt_path)

    # test EvalHook with customer test_fn and greater/less keys
    loader = DataLoader(EvalDataset())
    model = Model()
    data_loader = DataLoader(EvalDataset())

    eval_hook = EvalHook(
        data_loader,
        save_best='acc',
        test_fn=mock.MagicMock(return_value={}),
        greater_keys=[],
        less_keys=['acc'])

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        ckpt_path = osp.join(tmpdir, 'best_acc_epoch_6.pth')

        assert runner.meta['hook_msgs']['best_ckpt'] == ckpt_path
        assert osp.exists(ckpt_path)
        assert runner.meta['hook_msgs']['best_score'] == -3

    # test EvalHook with specified `out_dir`
    loader = DataLoader(EvalDataset())
    model = Model()
    data_loader = DataLoader(EvalDataset())
    out_dir = 's3://user/data'
    eval_hook = EvalHook(
        data_loader, interval=1, save_best='auto', out_dir=out_dir)

    with patch.object(PetrelBackend, 'put') as mock_put, \
         patch.object(PetrelBackend, 'remove') as mock_remove, \
         patch.object(PetrelBackend, 'isfile') as mock_isfile, \
         tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        basename = osp.basename(runner.work_dir.rstrip(osp.sep))
        ckpt_path = f'{out_dir}/{basename}/best_acc_epoch_4.pth'

        assert runner.meta['hook_msgs']['best_ckpt'] == ckpt_path
        assert runner.meta['hook_msgs']['best_score'] == 7

    assert mock_put.call_count == 3
    assert mock_remove.call_count == 2
    assert mock_isfile.call_count == 2


@patch('mmcv.engine.single_gpu_test', MagicMock)
@patch('mmcv.engine.multi_gpu_test', MagicMock)
@pytest.mark.parametrize('EvalHookParam', [EvalHook, DistEvalHook])
@pytest.mark.parametrize('_build_demo_runner,by_epoch',
                         [(_build_epoch_runner, True),
                          (_build_iter_runner, False)])
def test_start_param(EvalHookParam, _build_demo_runner, by_epoch):
    # create dummy data
    dataloader = DataLoader(EvalDataset())

    # 0.1. dataloader is not a DataLoader object
    with pytest.raises(TypeError):
        EvalHookParam(dataloader=MagicMock(), interval=-1)

    # 0.2. negative interval
    with pytest.raises(ValueError):
        EvalHookParam(dataloader, interval=-1)

    # 0.3. negative start
    with pytest.raises(ValueError):
        EvalHookParam(dataloader, start=-1)

    # 1. start=None, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=1, by_epoch=by_epoch)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 2. start=1, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(
        dataloader, start=1, interval=1, by_epoch=by_epoch)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 3. start=None, interval=2: perform evaluation after epoch 2, 4, 6, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=2, by_epoch=by_epoch)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 1  # after epoch 2

    # 4. start=1, interval=2: perform evaluation after epoch 1, 3, 5, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(
        dataloader, start=1, interval=2, by_epoch=by_epoch)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 3

    # 5. start=0, interval=1: perform evaluation after each epoch and
    #    before epoch 1.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=0, by_epoch=by_epoch)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    # 6. resuming from epoch i, start = x (x<=i), interval =1: perform
    #    evaluation after each epoch and before the first epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=1, by_epoch=by_epoch)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    if by_epoch:
        runner._epoch = 2
    else:
        runner._iter = 2
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # before & after epoch 3

    # 7. resuming from epoch i, start = i+1/None, interval =1: perform
    #    evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=2, by_epoch=by_epoch)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    if by_epoch:
        runner._epoch = 1
    else:
        runner._iter = 1
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 2 & 3


@pytest.mark.parametrize('runner,by_epoch,eval_hook_priority',
                         [(EpochBasedRunner, True, 'NORMAL'),
                          (EpochBasedRunner, True, 'LOW'),
                          (IterBasedRunner, False, 'LOW')])
def test_logger(runner, by_epoch, eval_hook_priority):
    loader = DataLoader(EvalDataset())
    model = Model()
    data_loader = DataLoader(EvalDataset())
    eval_hook = EvalHook(
        data_loader, interval=1, by_epoch=by_epoch, save_best='acc')

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_logger')
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        runner = EpochBasedRunner(
            model=model, optimizer=optimizer, work_dir=tmpdir, logger=logger)
        runner.register_logger_hooks(
            dict(
                interval=1,
                hooks=[dict(type='TextLoggerHook', by_epoch=by_epoch)]))
        runner.register_timer_hook(dict(type='IterTimerHook'))
        runner.register_hook(eval_hook, priority=eval_hook_priority)
        runner.run([loader], [('train', 1)], 1)

        path = osp.join(tmpdir, next(scandir(tmpdir, '.json')))
        with open(path) as fr:
            fr.readline()  # skip the first line which is `hook_msg`
            train_log = json.loads(fr.readline())
            assert train_log['mode'] == 'train' and 'time' in train_log
            val_log = json.loads(fr.readline())
            assert val_log['mode'] == 'val' and 'time' not in val_log
