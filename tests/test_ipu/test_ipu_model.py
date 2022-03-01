# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import random
import string
import tempfile

import pytest
import torch
import torch.nn as nn

# from mmcv.runner import build_runner
from mmcv.ipu import parse_ipu_options, build_from_cfg_with_wrapper,\
    IPU_MODE, ipu_model_wrapper, wrap_optimizer_hook,\
    IpuFp16OptimizerHook


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU6()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def test_build_model():
    ipu_options = parse_ipu_options({})
    model = TestModel()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    logger = logging.getLogger()
    modules_to_record = ['bn']
    pipeline_cfg = dict(
        train_split_edges=[
            dict(
                layer_to_call='conv',
                ipu_id=0),
        ])
    fp16_cfg = {'loss_scale': 0.5}
    ipu_model = ipu_model_wrapper(
                model, ipu_options, optimizer, logger,
                modules_to_record=modules_to_record, pipeline_cfg=pipeline_cfg,
                fp16_cfg=fp16_cfg)

test_build_model()