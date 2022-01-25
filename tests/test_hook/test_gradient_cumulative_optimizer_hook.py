import logging
import shutil
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmcv.runner import (GradientCumulativeOptimizerHook, OptimizerHook,
                         build_runner)


def test_gradient_cumulative_optimizer_hook():

    class ToyModel(nn.Module):

        def __init__(self, with_norm=False):
            super().__init__()
            self.fp16_enabled = False
            self.fc = nn.Linear(3, 2)
            nn.init.constant_(self.fc.weight, 1.)
            nn.init.constant_(self.fc.bias, 1.)
            self.with_norm = with_norm
            if with_norm:
                self.norm = nn.BatchNorm1d(2)

        def forward(self, x):
            x = self.fc(x)
            if self.with_norm:
                x = self.norm(x)
            return x

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).mean(), num_samples=x.shape[0])

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).mean(), num_samples=x.shape[0])

    def build_toy_runner(config=dict(type='EpochBasedRunner', max_epochs=3)):
        model = ToyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        tmp_dir = tempfile.mkdtemp()

        runner = build_runner(
            config,
            default_args=dict(
                model=model,
                work_dir=tmp_dir,
                optimizer=optimizer,
                logger=logging.getLogger(),
                meta=dict()))
        return runner

    with pytest.raises(AssertionError):
        # cumulative_iters only accepts int
        GradientCumulativeOptimizerHook(cumulative_iters='str')

    with pytest.raises(AssertionError):
        # cumulative_iters only accepts positive number
        GradientCumulativeOptimizerHook(cumulative_iters=-1)

    # test epoch based runner
    data = torch.rand((6, 3))
    # optimize with cumulative_iters
    loader_1 = DataLoader(data, batch_size=1)
    runner_1 = build_toy_runner()
    optimizer_hook = GradientCumulativeOptimizerHook(
        grad_clip=dict(max_norm=0.2), cumulative_iters=3)
    runner_1.register_hook(optimizer_hook)
    runner_1.run([loader_1], [('train', 1)])

    # optimize without cumulative_iters
    loader_2 = DataLoader(data, batch_size=3)
    runner_2 = build_toy_runner()
    optimizer_hook = OptimizerHook(grad_clip=dict(max_norm=0.2))
    runner_2.register_hook(optimizer_hook)
    runner_2.run([loader_2], [('train', 1)])

    # test optimizer works well
    assert (runner_1.model.fc.weight < 1).all()
    assert (runner_1.model.fc.bias < 1).all()
    # test optimizer with cumulative_iters gets the same results
    assert torch.allclose(runner_1.model.fc.weight, runner_2.model.fc.weight)
    assert torch.allclose(runner_1.model.fc.bias, runner_2.model.fc.bias)
    shutil.rmtree(runner_1.work_dir)
    shutil.rmtree(runner_2.work_dir)

    # test iter based runner
    data = torch.rand((8, 3))
    # optimize with cumulative_iters
    loader_1 = DataLoader(data, batch_size=1)
    runner_1 = build_toy_runner(dict(type='IterBasedRunner', max_iters=8))
    optimizer_hook = GradientCumulativeOptimizerHook(
        grad_clip=dict(max_norm=0.2), cumulative_iters=3)
    runner_1.register_hook(optimizer_hook)
    runner_1.run([loader_1], [('train', 1)])

    # optimize without cumulative_iters
    loader_2_divisible = DataLoader(data[:6], batch_size=3)
    loader_2_remainder = DataLoader(data[6:], batch_size=2)
    runner_2 = build_toy_runner(dict(type='IterBasedRunner', max_iters=3))
    optimizer_hook = OptimizerHook(grad_clip=dict(max_norm=0.2))
    runner_2.register_hook(optimizer_hook)
    runner_2.run([loader_2_divisible, loader_2_remainder], [('train', 2),
                                                            ('train', 1)])

    # test optimizer works well
    assert (runner_1.model.fc.weight < 1).all()
    assert (runner_1.model.fc.bias < 1).all()
    # test optimizer with cumulative_iters gets the same results
    assert torch.allclose(runner_1.model.fc.weight, runner_2.model.fc.weight)
    assert torch.allclose(runner_1.model.fc.bias, runner_2.model.fc.bias)
    shutil.rmtree(runner_1.work_dir)
    shutil.rmtree(runner_2.work_dir)

    # test has_batch_norm
    model = ToyModel(with_norm=True)
    optimizer_hook = GradientCumulativeOptimizerHook(
        grad_clip=dict(max_norm=0.2), cumulative_iters=3)
    assert optimizer_hook.has_batch_norm(model)
